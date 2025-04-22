# cuEmbed
This directory contains the core implementation of cuembed.

The library is divided into *embedding lookup* operations (found in `embedding_lookup.cuh`) and *index transformations* (found in `index_transformations.cuh`). 

## Embedding Lookup Operations
Embedding lookup operations (forward and backward) follow a similar computational pattern, primary consisting of three kinds of operations: 
-	Read the lookup indices. 
-	Read the corresponding input rows. 
-	Accumulate & write output. 

This embedding forward and backward implementation aims to achieve SOL memory bandwidth utilization by maximizing loads in flight when reading embedding rows. Launch parameters of the provided forward and backward kernels are determined by heuristics and the workload description.

`embedding_lookup.cuh` contains the following primary functions for forward and backward propagation. 

### Forward Propagation

The embedding forward operation accepts an embedding table in a dense row-major format, and lookup indicies in either a fixed-hotness format or compressed-sparse-row (CSR) format and processes them using one of three currently supported reduction modes: `CombineMode::kSum`, `CombineMode::kMean`, and `CombineMode::kConcat`.

```cpp
template <typename InputT,
          typename OutputT,
          typename IndexT,
          typename OffsetT,
          bool fp16_math>
void EmbeddingForward(const InputT* params,
                      const int embed_width,
                      const IndexT* indices,
                      const OffsetT* offsets,
                      const typename GetElemT<InputT>* weights,
                      const int batch_size,
                      const int num_hots,
                      const CombineMode mode,
                      OutputT* ret,
                      const cudaStream_t stream = 0) 
```
- For fixed hotness indices, `num_hots` indicates the hotness value, which is the same for every sample in the batch. `offsets` must be nullptr since there is no explicit offset array to be passed to the kernel.

- For CSR indices, `num_hots` must be 0. `offsets` points to the data of the explicit offset array indicating the starting point of indices for each sample in the batch. 

- For reduction type sum, `weights` can be nullptr to indicate plain reduction. Otherwise the weight of a specific lookup indice would be apply to the loaded rows before reduction. 

- When `fp16_math` is true, math operations (multiplication and summation) on fp16 embedding rows are performed in fp16.

__Parameters__
- __params__ Pointer to the embedding table data.
- __embed_width__ Number of elements in each embedding row. 
- __indices__ Pointer to the lookup indices.
- __offsets__ Pointer to the offsets (CSR format). Must be nullptr when launching for fixed hotness.
- __weights__ Pointer to the weight array. Weight for a specific lookup index is applied to the loaded embedding row before reduction. If nullptr, will use just the embedding row for reduction. The type for the weights must the be the same as the input type. If the input type is structured, then the user need to define their own `GetElemT<InputT>` specialization.
- __batch_size__  Batch size of the embedding lookup workload.
- __num_hots__ Number of rows to lookup for each sample in batch. Must be 0 when launching for CSR indices layout.
- __mode__ `ReductionType::kSum` (computes the summation of the looked up rows for each sample) or `ReductionType::kConcat` (concatenates all looked up rows).
- __ret__ Pointer to the output location.
- __stream__ Optional. The cudaStream to launch the kernel asynchronously. If not specified, will launch the kernel on default stream.

### Backward Propagation

The backward embedding operation accepts the incoming gradients and uses these along with the transposed indices to generate a gradient with respect to the embedding table. In the event of multiple indices pointing to the same embedding row, the gradients are summed, potentially using atomic operations.

EmbeddingBackward can produce either full embedding gradients or compressed embedding gradients. For dense embedding gradients, the output embedding gradient is the same size as the original embedding table, but will only will be modified in rows specified by the indices. For compressed embedding gradients, the output embedding gradient will have a number of rows which is equal to the number of unique lookup indices, and it will also produce an inverse mapping between the rows of the compressed gradient and the original row IDs in the embedding table.

```cpp
template <typename GradT, typename IndexT>
void EmbeddingBackward(const GradT* grad_y,
                       const int embed_width,
                       const int num_grad_embedding_rows,
                       const int nnz,
                       const IndexT* transpose_indices,
                       const IndexT* transpose_sample_ids,
                       const IndexT* transpose_remapped_indices,
                       const GradT* transpose_weights,
                       const bool skip_grad_init,
                       GradT* grad_embedding,
                       IndexT* inverse_mapping,
                       const cudaStream_t stream = 0)
```

- The inputs `transpose_indices`, `transpose_sample_ids`, and `transpose_weights` are indices in coordinate (COO) format, produced by *Transpose*. All repeating indices in `transpose_indices` must be grouped contiguously. 

- If `transpose_weights` is provided then these weights will be multiplied with the `y_grad` rows prior to accumulation. `transpose_weights` may be set to nullptr for the unweighted case.

- The input `transpose_remapped_indices` holds indices needed for compressed gradients, described below, and is produced by *ComputeCompressedGradIndices*.

- If full gradient is desired, then `transpose_remapped_indices` should be set to nullptr, `grad_embedding` should be allocated with `num_grad_embedding_rows` rows, which is equal to the total number of categories in this case, and `inverse_mapping` will not be written.

- If compressed gradient is desired, then `transpose_remapped_indices` should be provided (i.e. by *ComputeCompressedGradIndices*), `grad_embedding` should be allocated with `num_grad_embedding_rows` rows, which is equal to the number of unique lookup indices (i.e. transpose_remapped_indices.back()+1) in this case, and `inverse_mapping` should be allocated with `num_grad_embedding_rows` elements. 

- The output gradient `grad_embedding` will be initialized to zero prior to the backward lookup operation, unless `skip_grad_init` is set.

__Parameters__
- __grad_y__ Pointer to the incoming gradient.
- __embed_width__ Number of elements in each embedding row. 
- __num_grad_embedding_rows__ Number of rows in grad_embedding.
- __nnz__  Total number of indices in COO input.
- __transpose_indices__ Pointer to the transposed lookup indices.
- __transpose_sample_ids__ Pointer to the transposed sample IDs.
- __transpose_remapped_indices__ Pointer to the remapped lookup indices (i.e. from *ComputeCompressedGradIndices*), required only if computing compressed gradient.
- __transpose_weights__ Pointer to the weight array. Set to nullptr for unweighted.
- __skip_grad_init__ If true, skip zero-initializion of grad_embedding.
- __grad_embedding__ Pointer to the gradient wrt embedding table.
- __inverse_mapping__ Pointer to the table indices corresponding to each row in `grad_embedding`, produced only for compressed gradients.
- __stream__ Optional. The cudaStream to launch the kernel asynchronously. If not specified, will launch the kernel on default stream.


## Index Transformations

Index transformations are required to support common use cases of the forward and backward lookup kernels. For example, indices must be converted from the fixed-hotness or compressed-sparse-row (CSR) format used in the forward pass, into a transposed coordinate (COO) format for the backward pass. Additionally, computing compressed gradients during the backward pass requires one to generate a mapping between row ids in the compressed gradient and the row ids in the embedding table. 

Consider the common use-case of calling forward backward propagation, beginning with indices in CSR format. The lookup indices are stored in an array named `indices`, and ordered according to the sample IDs within the batch. (i.e. indices for sample ID 0, followed by sample IDs 1, 2, ..`batch_size-1`.) The `offsets` array contains a pointer to the beginning of each sample in the batch. The `offsets` array has size `batch_size+1`, with the last element containing the length of the `indices` array. When viewed as a sparse matrix, the forward CSR indices would look like this:

```
            <-- Embedding Categories-->
Sample    0: X         X           X
Sample    1:     X      
Sample    2: X              X
...
Sample bs-1:           X    X  

```

Note that the rows are samples, the columns are embedding categories, and the data is stored in row-major order. This is helpful to facilitate efficient row-wise reductions within a batch sample during the forward pass. 

However, for the backward pass we need to accumulate independenly all the gradients corresponding to a single embedding category (i.e. summing vertically in the above picture). In order to do this efficiently, we want the indices stored instead in a column-major or transposed ordering. A simple and performant way to transpose sparse matrices is to convert to coordinate format (i.e. three nnz-length arrays: rows, columns, and weights), and sort the arrays by either rows or columns

We provide the helper functions *ExtractRowIdsFromFixed*, *ExtractRowIdsFromCSR*, and *ExtractRowIdsForConcat* to help convert from the forward Fixed or CSR format to the explicit COO format. We also provide the *Transpose* function to reorder the COO indices by columns instead of rows. 


`index_transformations.cuh` contains the following functions:

### Conversion from Fixed-Hotness to COO
Produce a nnz-length `row_ids` array which has fixed offsets from 0 to batch_size, e.g.: `num_hots = 3 -> row_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]`
```cpp
template <typename IndexT>
void ExtractRowIdsFromFixed(const int batch_size,
                            const int num_hots,
                            IndexT* row_ids,
                            const cudaStream_t stream = 0);
```
### Conversion from CSR to COO
Produce a nnz-length `row_ids` array which has explicit offsets read from CSR, e.g.: `offsets : [0, 2, 3, 5] -> row_ids = [0, 0, 1, 2, 2]`
```cpp
template <typename IndexT, typename OffsetT>
void ExtractRowIdsFromCSR(const OffsetT* offsets,
                          const int batch_size,
                          IndexT* row_ids,
                          const cudaStream_t stream = 0);
```
### Conversion to COO for Concat
Produce a nnz-length `row_ids` array which has the sequence 0 .. nnz. e.g. `row_ids = [0, 1, 2, 3, ...]`
```cpp
template <typename IndexT>
void ExtractRowIdsForConcat(const int nnz, 
                            IndexT* row_ids,
                            const cudaStream_t stream = 0);
```
### Transpose

Reorders indices from sample-id-first ordering as is needed during forward to table-index-first ordering needed for backward. Output indices are produced in coordinate (COO) format.

```cpp
template <typename IndexT, typename WeightT>
void Transpose(const IndexT* rows,
               const IndexT* cols,
               const WeightT* weights,
               const int nnz,
               IndexT* transpose_rows,
               IndexT* transpose_cols,
               WeightT* transpose_weights,
               char* work,
               size_t* lwork,
               const cudaStream_t stream = 0);
```


- Input `rows`, `cols`, `weights` are the indices in COO format. For the embedding use case, `rows` contains the sample IDs during forward pass and `cols` contains the embedding lookup indices. 

- Output is stored in output arrays `transpose_rows`, `transpose_cols` and `transpose_weights` should be allocated with `nnz` elements. 

- If input `weights` are set to nullptr, then output `transpose_weights` will not be set. 

- For the embedding use case, `transpose_rows` contains the embedding lookup indices which are now in sorted order. `transpose_cols` and `transpose_weights` contain the sample IDs and optionally the weights corresponding to the reordered rows. 

- The function should first be called with `work` set to nullptr to perform a workspace query. The required size of `work` array in bytes will be returned in `lwork`. Then the function should be called a second time with `work` pointing to allocated workspace of size `lwork`. 

__Parameters__
- __rows__ Pointer to the lookup indices.
- __cols__ Pointer to the offsets (CSR format) used during forward. Must be nullptr when launching for fixed hotness.
- __weights__ Pointer to the weight array used during forward. If nullptr, will not produce transposed weights. 
- __nnz__ Number of nonzeros.
- __transpose_rows__ Pointer to the output transposed table indices.
- __transpose_cols__ Pointer to the output transposed sparse indices.
- __transpose_weights__ Pointer to the transposed weight array. If input weights is nullptr, then will not produce transposed weights. 
- __work__ Pointer to scratch workspace. Set to nullptr for workspace query.
- __lwork__ Pointer to size of scratch workspace.
- __stream__ Optional. The cudaStream to launch the kernel asynchronously. If not specified, will launch the kernel on default stream.



### Compressed Gradient Index Conversion

In some cases, the number of embedding rows actually referenced by the indices is much smaller than the total number of rows. In these cases, it may be advantageous to produce a compressed gradient which stores only the nonzero rows of the embedding gradient. However, this requires an additional step of remapping the indices from dense embedding row ids to compressed embedding row ids. This process is pictured below. The indices which are initially distributed between 0 and `num_categories` values, are remapped to the range of 0 and `num_unique`, e.g. `indices =  [4, 4, 7, 8, 8, 8, 18] -> remapped_indices = [0, 0, 1, 2, 2, 2, 3]`

We provide the helper function *ComputeCompressedGradIndices* to do the above transformation. Note that the value `num_unique` can be attained from remapped_indices.back() + 1 after calling this function.

```cpp
template <typename IndexT>
void ComputeCompressedGradIndices(const IndexT* indices,
                              const int nnz,
                              IndexT* remapped_indices,
                              char* work,
                              size_t* lwork,
                              const cudaStream_t stream = 0) 
```

- The function should first be called with `work` set to nullptr to perform a workspace query. The required size of `work` array in bytes will be returned in `lwork`. Then the function should be called a second time with `work` pointing to allocated workspace of size `lwork`. 

__Parameters__
- __indices__ Pointer to the lookup indices, grouped by index.
- __nnz__ Length of the indices array.
- __remapped_indices__ Pointer to the remapped lookup indices (output)
- __work__ Temporary workspace
- __lwork__ Size of workspace in bytes (input/output)
- __stream__ Optional. The cudaStream to launch the kernel asynchronously. If not specified, will launch the kernel on default stream.


