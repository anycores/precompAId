#include <iostream>
#include "xg_runtime_api.h"


void test_whisper(const std::string& weight_path, const std::string& input_path);

int main(int argc, char** argv) {

	if (argc == 3)
	{
		std::string weight_path = argv[1];
	    std::string input_path = argv[2];
	    test_whisper(weight_path, input_path);
	}

	return 0;
}

void test_whisper(const std::string& weight_path, const std::string& input_path)
{
	XgModelInfo minfo = {};
	xg_get_model_info(&minfo);
	std::cout << minfo.model_name << " " << minfo.model_version << std::endl;
    
	std::cout << "initing graph" << std::endl;
	XgGraph* graph = nullptr;
	if (xg_init_graph(weight_path, XGWeightSource::XG_ONNX, &graph) != XGResult::XG_SUCCESS)
	{
		std::cout << "Graph init error" << std::endl;
		return;
	}
	else
	{
		std::cout << "Graph init: successful" << std::endl;
	}

	XgData* input_data = nullptr;
	if (xg_allocate_input_compatible_data(0, &input_data) != XGResult::XG_SUCCESS)
	{
		std::cout << "Input allocation error" << std::endl;
		return;
	}
	else
	{
		std::cout << "Input allocation: successful" << std::endl;
	}
        
	// load the data into XgData
	reinterpret_cast<std::string*>(input_data->raw_data)[0] = input_path;

	if (xg_set_input_data(graph, 0, input_data) != XGResult::XG_SUCCESS)
	{
		std::cout << "Input data set error" << std::endl;
		return;
	}
	else
	{
		std::cout << "Input data set: successful" << std::endl;
	}
    
	// execute the graph
	xg_execute_graph(graph);

	// write output
	XgData* output_data = nullptr;
	if (xg_get_output_data(graph, 0, &output_data) != XGResult::XG_SUCCESS)
	{
		std::cout << "Getting output error" << std::endl;
		return;
	}
	else
	{
		std::cout << "Getting output: successful" << std::endl;
	}

	// print output
	std::string* o1 = reinterpret_cast<std::string*>(output_data->raw_data);
	std::cout << o1[0] << std::endl;
}
