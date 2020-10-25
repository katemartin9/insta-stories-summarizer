from files_ops import get_stories, check_file_type, convert_video_to_np


if __name__ == '__main__':
    folder_name = 'insta_stories'
    insta_dict = get_stories(folder_name)
    for key, vals in insta_dict.items():
        for val in vals:
            if check_file_type(val):
                print(f'{key}: {convert_video_to_np(val).shape}')


