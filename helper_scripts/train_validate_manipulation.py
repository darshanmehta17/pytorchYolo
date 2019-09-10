if __name__ == '__main__':
    new_path = '/home/mitchell/AMP_YOLO_data/train_data/images_subset'
    txt_path = '/home/mitchell/pytorchYolo/cfg/train_test_locs'
    reader = open(txt_path + '/train_amp_rename.txt', 'r')
    writer = open(txt_path + '/train_amp_rename_subset.txt', 'w+')
    
    for line in reader:
        #print(line)
        save_name = new_path + '/' + line.split('/')[-1]
        print(save_name)
        writer.write(save_name)
    reader.close()
    writer.close()    
    reader = open(txt_path + '/val_amp_rename.txt', 'r')
    writer = open(txt_path + '/val_amp_rename_subset.txt', 'w+')
    
    for line in reader:
        line.split('/')
        save_name = new_path +  '/' + line.split('/')[-1]
        writer.write(save_name)
    reader.close()
    writer.close()        