#ifndef __XG_RUNTIME_API__
#define __XG_RUNTIME_API__

#include <vector>
#include <string>

#if _WIN32
    #define XG_API extern "C" __declspec(dllexport)
#elif __unix__ || __linux__
    #define XG_API extern "C"
#endif

// XG type definitions

enum class XGResult
{
	XG_SUCCESS,
	XG_INPUT_SIZE_MISSMATCH,
	XG_INPUT_TYPE_MISSMATCH,
	XG_WRONG_INPUT_INDEX,
	XG_WRONG_OUTPUT_INDEX,
	// device related
	XG_DEVICE_NOT_SUPPORTED,
	XG_MEMORY_ALLOCATION_FAILED,
	// weight file access
	XG_FILE_NOT_FOUND,
	XG_EXECUTION_FAILED
};

enum class XGWeightSource
{
	XG_ONNX,
	XG_XGDB
};

enum class XGDataType
{
	XG_BOOL,
	XG_TOKEN,
	XG_STRING,
	XG_UINT8,
	XG_UINT16,
	XG_UINT32,
	XG_UINT64,
	XG_INT8,
	XG_INT16,
	XG_INT32,
	XG_INT64,
	XG_BFLOAT16,
	XG_FLOAT16,
	XG_FLOAT32,
	XG_FLOAT64
};

// access information about the contained model
struct XgModelInfo
{
	std::string model_name;
	std::string model_version;
	std::string device;    // cpu, gpu, tpu etc.
	std::string hardware;  // e.g. intel i7 9th gen
	unsigned int num_inputs;
	unsigned int num_outputs;
};

XG_API void xg_get_model_info(
	XgModelInfo* model_info
);

XG_API bool is_current_device_supported();  // may be list the supported devices on this machine

// create graph
struct XgGraph;

XG_API XGResult xg_init_graph(
	const std::string& weight_path, 
	const XGWeightSource weight_source,
	XgGraph** graph
);
XG_API XGResult xg_execute_graph(
	XgGraph* graph
);
XG_API XGResult xg_destroy_graph(
	XgGraph** graph
);

// set the input to the graph,
// query the output

struct XgData
{
	XGDataType dtype;
	unsigned int size_in_bytes;
	unsigned int dimension;
	unsigned int length;
	unsigned int* shape;
	char* raw_data;
};

XG_API unsigned int xg_calculate_tensor_size_in_bytes(
	const XGDataType dtype,
	const unsigned int* shape,
	const unsigned int dimension
);
XG_API XGResult xg_allocate_input_compatible_data(
	const unsigned int input_idx,
	XgData** data
);
XG_API XGResult xg_destroy_data(
	XgData** data
);
XG_API XGResult xg_get_output_data(
	const XgGraph* graph,
	const unsigned int output_idx,
	XgData** data
);
XG_API XGResult xg_set_input_data(
	const XgGraph* graph,
	const unsigned int input_idx,
	const XgData* data
);

// helper functions
XG_API bool xg_is_data_bool(const XgData* data);
XG_API bool xg_is_data_uint8(const XgData* data);
XG_API bool xg_is_data_uint16(const XgData* data);
XG_API bool xg_is_data_uint32(const XgData* data);
XG_API bool xg_is_data_uint64(const XgData* data);
XG_API bool xg_is_data_int8(const XgData* data);
XG_API bool xg_is_data_int16(const XgData* data);
XG_API bool xg_is_data_int32(const XgData* data);
XG_API bool xg_is_data_int64(const XgData* data);
XG_API bool xg_is_data_bfloat16(const XgData* data);
XG_API bool xg_is_data_float16(const XgData* data);
XG_API bool xg_is_data_float32(const XgData* data);
XG_API bool xg_is_data_float64(const XgData* data);

#endif  // __XG_RUNTIME_API__
