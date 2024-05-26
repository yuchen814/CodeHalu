import os
import json
from tqdm import tqdm

def load_problems(problems_root, return_dict=False):
    subdirs = [
        os.path.join(problems_root, d) for d in os.listdir(problems_root) 
            if os.path.isdir(os.path.join(problems_root, d))
    ]
    problems = []
    
    for subdir in tqdm(sorted(subdirs)):  # Maintain the same order
        data_json_path = os.path.join(subdir, 'data.json')
        images_folder = os.path.join(subdir, 'images')
        problem_id = os.path.basename(subdir)

        # Read data.json
        with open(data_json_path, 'r') as file:
            problem_data = json.load(file)

            # Load images from 'images' folder
            images = []
            image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.png')],
                                    key=lambda x: int(os.path.splitext(x)[0]))
            for img_file in image_files:
                img_path = os.path.join(images_folder, img_file)
                # image = Image.open(img_path).convert("RGBA")
                # new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
                # new_image.paste(image, (0, 0), image)
                # new_image = new_image.convert('RGB')
                # del image
                # images.append(new_image)
                image = Image.open(img_path).convert('RGB')
                images.append(image)
            
            problems.append( 
                {
                    'problem_id': problem_id,
                    'problem': problem_data,
                    'images': images,
                }
            )
    
    if return_dict:
        return {p['problem_id']: p for p in problems}
    else:
        return problems