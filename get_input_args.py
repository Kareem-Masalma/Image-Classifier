import argparse
 
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("saved_model", type=str, help="Path to the saved Keras model.")
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K predictions.")
    parser.add_argument("--category_names", type=str, default=None,
                        help="Path to JSON file mapping labels to flower names.")
    
    return parser.parse_args()