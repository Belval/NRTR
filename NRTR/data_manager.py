import re
import os
import numpy as np
import config

from utils import resize_image, label_to_array, label_to_array_2

class DataManager(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_test_ratio):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.max_char_count = 25
        self.data, self.data_len = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset
        self.train_batches = self.__generate_all_train_batches()
        self.test_batches = self.__generate_all_test_batches()

    def __load_data(self):
        """
            Load all the images in the folder
        """

        print('Loading data')

        examples = []

        count = 0
        skipped = 0
        for i, f in enumerate(os.listdir(self.examples_path)):
            if i > 100000:
                break
            if len(f.split('_')[0]) > self.max_char_count:
                continue
            arr, initial_len = resize_image(
                os.path.join(self.examples_path, f),
                self.max_image_width
            )
            examples.append(
                (
                    arr,
                    f.split('_')[0].lower(),
                    label_to_array(f.split('_')[0].lower()),
                    label_to_array_2(f.split('_')[0].lower())
                )
            )
            count += 1

        print(count)

        return examples, len(examples)

    def __generate_all_train_batches(self):
        train_batches = []
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la, raw_batch_la_2 = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_seq_len = np.reshape(
                [len(y) for y in raw_batch_y],
                (-1)
            )

            batch_dt = np.reshape(
                np.array(raw_batch_la),
                (-1, 25, config.NUM_CLASSES)
            )

            batch_dt_2 = np.reshape(
                np.array(raw_batch_la_2),
                (-1, 25)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.max_image_width, 32, 1)
            )

            train_batches.append((batch_y, batch_seq_len, batch_dt, batch_dt_2, batch_x))
        return train_batches

    def __generate_all_test_batches(self):
        test_batches = []
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset

            raw_batch_x, raw_batch_y, raw_batch_la, _ = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_dt = np.zeros(
                np.shape(
                    np.reshape(
                        np.array(raw_batch_la),
                        (-1, 25, config.NUM_CLASSES)
                    )
                )
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.max_image_width, 32, 1)
            )

            test_batches.append((batch_y, batch_dt, batch_x))
        return test_batches
