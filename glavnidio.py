import cv2
import os
import scipy.io
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import argparse

def adjust_brightness(image, factor=1.2):
    brightened_image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return brightened_image

def preprocess_image(image_path, print_processing=True):
    try:
        if print_processing:
            print(f"Processing image: {image_path}")

        folder_path = 'imdb_crop/imdb_crop'
        image_path_2 = os.path.join(folder_path, image_path)
        img = cv2.imread(image_path_2)

        if img is None or img.size == 0:
            raise FileNotFoundError(f"Error: Unable to read or empty image at {image_path}")
            return None

        if np.isnan(img).any():
            raise ValueError(f"Error: NaN values in image at {image_path}")
            return None

        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error processing image at {image_path}: {str(e)}")
        return None

def create_gender_model():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def create_age_model():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model

def main(train_mode=True):
    mat_file_path = "imdb_crop/imdb_crop/imdb.mat"

    mat_contents = scipy.io.loadmat(mat_file_path)

    dataset_root = "imdb_crop/imdb_crop"

    all_folders = [folder for folder in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, folder))]

    train_folders, test_folders = train_test_split(all_folders, test_size=0.2, random_state=42)

    imdbMat = scipy.io.loadmat('imdb_crop/imdb_crop/imdb.mat')
    imdbPlace = imdbMat['imdb'][0][0]
    place = imdbPlace
    where = 'imdb_crop'

    features = []
    labels_gender = []
    labels_age = []

    total_samples = 20000
    if train_mode:
        
        split_index = int(0.8 * total_samples)
    else:
        
        split_index = total_samples - int(0.8 * total_samples)

    for i in range(total_samples):
        bYear = int(place[0][0][i] / 365)
        taken = place[1][0][i]
        path = place[2][0][i][0]
        gender = place[3][0][i]
        age = taken - bYear
        img = preprocess_image(path, print_processing=train_mode)

        if img is not None and gender is not None and age is not None:
            features.append(img)
            labels_gender.append(gender)
            labels_age.append(age)

        if i >= split_index:
            break

    features = np.array(features)
    labels_gender = np.array(labels_gender)
    labels_age = np.array(labels_age)

    nan_labels_gender = np.isnan(labels_gender)
    nan_labels_age = np.isnan(labels_age)
    nan_labels = np.logical_or(nan_labels_gender, nan_labels_age)

    if np.any(nan_labels):
        print(f"Found NaN values in labels. Indices with NaN: {np.where(nan_labels)}")

        total_samples_before = len(labels_gender)
        nan_samples_before = np.sum(nan_labels)
        data_loss_percentage_before = (nan_samples_before / total_samples_before) * 100

        print(f"Total samples before: {total_samples_before}")
        print(f"NaN samples before: {nan_samples_before}")
        print(f"Data loss percentage before: {data_loss_percentage_before:.2f}%")

        features = features[~nan_labels]
        labels_gender = labels_gender[~nan_labels]
        labels_age = labels_age[~nan_labels]

        np.savez("cleaned_dataset.npz", features=features, labels_gender=labels_gender, labels_age=labels_age)

    if train_mode:
        X_train = features[:split_index]
        y_train_gender = labels_gender[:split_index]
        y_train_age = labels_age[:split_index]

        print(features.dtype)
        print(labels_gender.dtype)
        print(labels_age.dtype)

        gender_model = create_gender_model()
        age_model = create_age_model()

        
        gender_model.fit(
            X_train,
            y_train_gender,
            epochs=10,
            batch_size=2
        )

        
        age_model.fit(
            X_train,
            y_train_age,
            epochs=10,
            batch_size=2
        )

       
        gender_model.save('gender_model.h5')
        age_model.save('age_model.h5')

        print("Models saved successfully.")
    else:
        try:
            
            gender_model = load_model('gender_model.h5')
            print("Gender model loaded successfully.")

            
            age_model = load_model('age_model.h5')
            print("Age model loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {str(e)}")

        
        test_loss_gender, test_acc_gender = gender_model.evaluate(features, labels_gender)
        print(f"Test Gender Accuracy: {test_acc_gender * 100:.2f}%")
        print(f"Number of samples tested: {len(features)}")

        
        test_loss_age, test_acc_age = age_model.evaluate(features, labels_age)
        print(f"Test Age Accuracy: {100 - test_acc_age:.2f}%")
        print(f"Number of samples tested: {len(features)}")

        
        gender_model = create_gender_model()
        age_model = create_age_model()

        gender_model.load_weights('gender_model.h5')
        age_model.load_weights('age_model.h5')

        
        num_examples = 5
        test_indices = np.random.choice(len(features), num_examples, replace=False)
        for i, test_index in enumerate(test_indices):
            example_image = features[test_index]
            example_gender_label = labels_gender[test_index]
            example_age_label = labels_age[test_index]

            predicted_gender_label = gender_model.predict(np.expand_dims(example_image, axis=0))
            predicted_gender_label = round(predicted_gender_label[0][0])

            predicted_age_label = age_model.predict(np.expand_dims(example_image, axis=0))
            predicted_age_label = round(predicted_age_label[0][0])

            
            resized_image = cv2.resize((example_image * 255).astype(np.uint8), (150, 150)) 
            cv2.imshow(f"Example {i + 1}", cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"Example {i + 1}:")
            print(f"  True Gender Label: {example_gender_label}")
            print(f"  Predicted Gender Label: {predicted_gender_label}")

            print(f"  True Age Label: {example_age_label}")
            print(f"  Predicted Age Label: {predicted_age_label}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the gender classification model.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    args = parser.parse_args()

    main(train_mode=args.train)