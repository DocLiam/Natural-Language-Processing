void train(double min_diff, double learning_rate, int cycles, int stream_train, int shift_count_train, int line_count_train, double *input_values_train, double *target_values_train, int stream_validate, int shift_count_validate, int line_count_validate, double *input_values_validate, double *target_values_validate, int *activation_values, int *hidden_sizes, int layer_count, int bias_count, int hidden_count, int weight_count, double *weight_values);
void test(int stream, int shift_count, int line_count, double *input_values, double *output_values, int *activation_values, int *hidden_sizes, int layer_count, int bias_count, int hidden_count, int weight_count, double *weight_values);