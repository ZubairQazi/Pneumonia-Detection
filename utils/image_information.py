import os
from PIL import Image

def get_image_stats(folder_path):
    total_width = 0
    total_height = 0
    total_images = 0
    largest_width = 0
    largest_height = 0
    smallest_width = float('inf')
    smallest_height = float('inf')

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            width, height = image.size
            total_width += width
            total_height += height
            total_images += 1
            largest_width = max(largest_width, width)
            largest_height = max(largest_height, height)
            smallest_width = min(smallest_width, width)
            smallest_height = min(smallest_height, height)

    if total_images == 0:
        return 0, 0, 0, 0, 0, 0  # No images found in the folder

    average_width = total_width / total_images
    average_height = total_height / total_images

    return (
        average_width, average_height,
        largest_width, largest_height,
        smallest_width, smallest_height
    )


if __name__ == '__main__':
    folder_path = input('Enter data path: ')
    
    average_width, average_height, \
        largest_width, largest_height, \
            smallest_width, smallest_height = get_image_stats(folder_path)

    print(f"Average Image Size: {average_width} x {average_height}")
    print(f"Largest Image Size: {largest_width} x {largest_height}")
    print(f"Smallest Image Size: {smallest_width} x {smallest_height}")
