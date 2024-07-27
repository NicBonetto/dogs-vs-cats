import os, shutil
from keras.preprocessing.image import ImageDataGenerator

def organize():

    # Raw images downloaded from Kaggle dataset
    original_dataset_dir = os.environ['DATASET_DIR']
    
    # Base directory for organized datasets
    base_dir = os.environ['BASE_DIR']

    if(!os.path.exists(base_dir)):
        os.mkdir(base_dir)
        
        # Organize the directories
        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)
        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)
        test_dir = os.path.join(base_dir, 'test')
        os.mkdir(test_dir)
        
        train_dogs_dir = os.path.join(train_dir, 'dogs')
        os.mkdir(train_dogs_dir)
        train_cats_dir = os.path.join(train_dir, 'cats')
        os.mkdir(train_cats_dir)
        
        validate_dogs_dir = os.path.join(validation_dir, 'dogs')
        os.mkdir(validate_dogs_dir)
        validate_cats_dir = os.path.join(validation_dir, 'cats')
        os.mkdir(validate_cats_dir)
        
        test_dogs_dir = os.path.join(test_dir, 'dogs')
        os.mkdir(test_dogs_dir)
        test_cats_dir = os.path.join(test_dir, 'cats')
        os.mkdir(test_cats_dir)
    
        # Copy first 5000 dog images into training dir
        fnames = ['{}.jpg'.format(i) for i in range(0, 5000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, 'Dog', fname)
            dest = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dest)
    
    
        # Copy first 5000 cat images into training dir
        fnames = ['{}.jpg'.format(i) for i in range(0, 5000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, 'Cat', fname)
            dest = os.path.join(train_cat_dir, fname)
            shutil.copyfile(src, dest)
    
        # Copy next 2500 dog images into validation dir
        fnames = ['{}.jpg'.format(i) for i in range(5000, 7500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, 'Dog', fname)
            dest = os.path.join(validate_dogs_dir, fname)
            shutil.copyfile(src, dest)
    
        # Copy next 2500 cat images into validation dir
        fnames = ['{}.jpg'.format(i) for i in range(5000, 7500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, 'Cat', fname)
            dest = os.path.join(validate_cats_dir, fname)
            shutil.copyfile(src, dest)
    
    
        # Copy next 2500 dog images into test dir
        fnames = ['{}.jpg'.format(i) for i in range(7500, 10000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, 'Dog', fname)
            dest = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dest)
    
        # Copy next 2500 cat images into test dir
        fnames = ['{}.jpg'.format(i) for i in range(7500, 10000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, 'Cat', fname)
            dest = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dest)


def process_images(image_dir):
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        image_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return generator
