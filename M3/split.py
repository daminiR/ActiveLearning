import os
import numpy as np
import shutil
root = "D:/c3"
for idx , (dirpath, dirnames, filenames) in enumerate(os.walk(root)):
    if idx != 0:
        print(dirpath.split('\\')[0])
        try:
            os.mkdir(os.path.join('D:/test10/', dirpath.split('\\')[1]))
        except FileExistsError:
            # directory already exists
            pass

        random_files = np.random.choice(filenames, int(len(filenames) * .2))
        # print(random_files)
        # print(random_files)
        for file in random_files:
            print(file)
            label = dirpath.split('c3\\')[1]
            src =  os.path.join(dirpath, file)
            dest = os.path.join('D:/test10/', label)
            print("dest")
            print(dest)
            # if not os.path.exists(os.path.join('D:/test10', dirpath.split('\\')[0])):

            if os.path.isfile(os.path.join(dirpath, file)):
                shutil.copy(src, dest)
                os.remove(os.path.join(dirpath, file))
            else:
                pass

