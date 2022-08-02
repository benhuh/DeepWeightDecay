import os
data_config = {}

task_list = os.listdir("/home/zipingxu/meta-dataset/data/data_source/omniglot/images_background")
data_config['omniglot'] = {
    'root_dir': "/home/zipingxu/meta-dataset/data/data_source/omniglot/images_background",
    'task_list': task_list,
    'num_task': len(task_list)
}
