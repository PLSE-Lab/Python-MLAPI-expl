import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Dense, Input, concatenate, Flatten, Dropout, BatchNormalization
from keras.optimizers import adam
from keras import losses
from keras import metrics


def load_data():
	train_data = pd.read_csv("../input/avito-demand-prediction/train.csv", parse_dates=["activation_date"])
	test_data = pd.read_csv("../input/avito-demand-prediction/test.csv", parse_dates=["activation_date"])

	train_data['activation_date'] = np.expand_dims(
		pd.to_datetime(train_data['activation_date']).dt.weekday.astype(np.int32).values, axis=-1)
	test_data['activation_date'] = np.expand_dims(
		pd.to_datetime(test_data['activation_date']).dt.weekday.astype(np.int32).values, axis=-1)

	y_col = 'deal_probability'

	return train_data, test_data, y_col


def fill_na(train_data, test_data):
	train_data['image_top_1'].fillna(value=3067, inplace=True)
	test_data['image_top_1'].fillna(value=3067, inplace=True)

	train_data['item_seq_number'].fillna(value=-1, inplace=True)
	test_data['item_seq_number'].fillna(value=-1, inplace=True)

	train_data['price'].fillna(value=-1, inplace=True)
	test_data['price'].fillna(value=-1, inplace=True)

	train_data['param_1'].fillna(value='_NA_', inplace=True)
	test_data['param_1'].fillna(value='_NA_', inplace=True)

	train_data['param_2'].fillna(value='_NA_', inplace=True)
	test_data['param_2'].fillna(value='_NA_', inplace=True)

	train_data['param_3'].fillna(value='_NA_', inplace=True)
	test_data['param_3'].fillna(value='_NA_', inplace=True)
	return train_data, test_data


def drop_unwanted(train_data, test_data):
	train_data.drop(['item_id', 'user_id', 'image'], axis=1, inplace=True)
	test_data.drop(['item_id', 'user_id', 'image'], axis=1, inplace=True)
	return train_data, test_data


def encode_cat_columns(all_data):
	cat_cols = ['region', 'parent_category_name', 'category_name', 'city', 'user_type', 'image_top_1', 'param_1',
	            'param_2', 'param_3']

	le_encoders = {x: LabelEncoder() for x in cat_cols}
	label_enc_cols = {k: v.fit_transform(all_data[k]) for k, v in le_encoders.items()}

	return le_encoders, label_enc_cols


def transform_title_description(train_data, test_data):
	train_data['title'] = train_data.title.apply(lambda x: len(str(x).split(' ')))
	train_data['description'] = train_data.description.apply(lambda x: len(str(x).split(' ')))

	test_data['title'] = test_data.title.apply(lambda x: len(str(x).split(' ')))
	test_data['description'] = test_data.description.apply(lambda x: len(str(x).split(' ')))
	return train_data, test_data


def scale_num_cols(train_data, test_data):
	stdScaler = StandardScaler()
	train_data[['price', 'item_seq_number', 'title', 'description']] = stdScaler.fit_transform(train_data[['price', 'item_seq_number', 'title', 'description']])
	test_data[['price', 'item_seq_number', 'title', 'description']] = stdScaler.fit_transform(test_data[['price', 'item_seq_number', 'title', 'description']])
	return train_data, test_data


def load_VGG16_img_features():
	train_img_features = sparse.load_npz('../input/vgg16-train-features/features.npz')
	test_img_features = sparse.load_npz('../input/vgg16-test-features/features.npz')
	return train_img_features, test_img_features


def split_train_validation(train_data):
	val_split = 0.15
	val_ix = int(np.rint(len(train_data) * (1. - val_split)))

	t_split_df = train_data[:val_ix]
	v_split_df = train_data[val_ix:]

	image_t_split_df = train_img_features[:val_ix]
	image_v_split_df = train_img_features[val_ix:]

	return t_split_df, v_split_df, image_t_split_df, image_v_split_df


def gen_samples(in_df, img_df, batch_size, loss_name):

	samples_per_epoch = in_df.shape[0]
	number_of_batches = samples_per_epoch / batch_size
	counter = 0

	while True:

		if batch_size == 0:
			out_df = in_df

		else:
			sub_img_frame = pd.DataFrame(img_df[batch_size * counter:batch_size * (counter + 1)].todense())

			sub_img_frame.columns = ['img_' + str(col) for col in sub_img_frame.columns]

			out_df = in_df[batch_size * counter:batch_size * (counter + 1)]

			for col in sub_img_frame.columns:
				out_df.insert(len(out_df.columns), col, pd.Series(sub_img_frame[col].values, index=out_df.index))

		feed_dict = {col_name: le_encoders[col_name].transform(out_df[col_name].values) for col_name in cat_cols}

		cont_cols = [x for x in out_df.columns if 'img_' in x]
		cont_cols.extend(['price', 'item_seq_number', 'title', 'description'])
		feed_dict['continuous'] = out_df[cont_cols].values

		counter += 1

		yield feed_dict, out_df[loss_name].values

		if counter <= number_of_batches:
			counter = 0


def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))


def build_model(label_enc_cols):
	all_embeddings, all_inputs = [], []

	for key, val in label_enc_cols.items():
		in_val = Input(shape = (1,), name = key)
		all_embeddings += [Flatten()(Embedding(val.max() + 1, (val.max() + 1) // 2)(in_val))]
		all_inputs += [in_val]

	concat_emb_layer = concatenate(all_embeddings)
	bn_emb = BatchNormalization()(concat_emb_layer)
	emb_layer = Dense(16, activation='relu')(Dropout(0.5)(bn_emb))

	cont_input = Input(shape = (516,), name = 'continuous')
	bn_cont = BatchNormalization()(cont_input)
	cont_feature_layer = Dense(16, activation = 'relu')(Dropout(0.5)(bn_cont))

	full_concat_layer = concatenate([emb_layer, cont_feature_layer])
	full_reduction = Dense(16, activation = 'relu')(full_concat_layer)

	out_layer = Dense(1, activation = 'sigmoid')(full_reduction)
	model = Model(inputs = all_inputs + [cont_input], outputs = [out_layer])

	return model



train_data, test_data, y_col = load_data()
train_data, test_data = fill_na(train_data, test_data)
train_data, test_data = drop_unwanted(train_data, test_data)
train_data, test_data = transform_title_description(train_data, test_data)

all_data = pd.concat([train_data, test_data], sort = False)
le_encoders, label_enc_cols = encode_cat_columns(all_data)

train_data, test_data = scale_num_cols(train_data, test_data)

train_img_features, test_img_features = load_VGG16_img_features()

t_split_df, v_split_df, image_t_split_df, image_v_split_df = split_train_validation(train_data)

model = build_model(label_enc_cols)

optimizer = optimizers.Adam(lr = 0.0005, beta_1 = 0.9, beta_2 = 0.999, epsilon = 0.1, decay = 0.0, amsgrad = False)
model.compile(optimizer = optimizer, loss = root_mean_squared_error, metrics = [root_mean_squared_error])

checkpoint = ModelCheckpoint('best_weights.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)
early = EarlyStopping(patience = 2, mode = 'min')

batch_size = 64
model.fit_generator(gen_samples(t_split_df, image_t_split_df, batch_size, y_col),
                    epochs = 1,
                    steps_per_epoch = t_split_df.shape[0] / batch_size,
                    validation_data = next(gen_samples(v_split_df, image_v_split_df, 128, y_col)),
                    validation_steps = 10,
                    callbacks = [checkpoint, early])

test_vars, test_id = next(gen_samples(test_data, test_img_features, test_data.shape[0], loss_name = ''))
model.load_weights('best_weights.hdf5')
preds = model.predict(test_vars)

subm = pd.read_csv("../input/avito-demand-prediction/sample_submission.csv")
subm['deal_probability'] = preds
subm.to_csv('submission_adam.csv', index = False)