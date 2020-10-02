
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from imgaug import augmenters as iaa
import imgaug as ia
import keras
import matplotlib.pyplot as plt
from time import perf_counter
import imageio
from sklearn import metrics
from skimage import morphology as skmorphology
from sklearn.utils import class_weight
import re




class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, images_paths, target_paths,
	             image_dimensions=(64, 64, 1), batch_size=64,
	             shuffle=False, augment=False):
		self.target_paths = target_paths
		self.images_paths = images_paths
		self.dim = image_dimensions
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.augment = augment
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.images_paths) / self.batch_size))

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.images_paths))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		'Generate one batch of data'
		# Selects indices of data for next batch
		indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
		images = np.array([plt.imread(self.images_paths[k]) for k in indexes], dtype=np.uint8)

		if self.target_paths is None:
			targets = np.array([])
		else:
			targets = np.array([plt.imread(self.target_paths[k]) for k in indexes], dtype=np.uint8) / 255

		if self.augment == True:
			images, targets = self.augmentor(images, targets)

			# ---- debug input image / label pairs ----
			#for (img, lbl) in zip(images, targets):
			#	print(img.shape, img.min(), img.max(), img.dtype)
			#	print(lbl.shape, lbl.min(), lbl.max(), img.dtype)
			#	fig, ax = plt.subplots(1, 2, figsize=(6, 3))
			#	ax[0].imshow(img, cmap='gray')
			#	ax[1].imshow(lbl, cmap='gray')
			#	plt.show()
			#	input()

		images = images.astype(np.float32) / 255.
		#images = (images - images.mean()) / images.std()

		return np.reshape(images, (*images.shape, self.dim[-1])),\
		       np.reshape(targets, (*targets.shape, 1))


	def augmentor(self, images, targets):
		sometimes = lambda aug: iaa.Sometimes(0.5, aug)
		seq = iaa.Sequential([
				iaa.Fliplr(0.5, name="Fliplr"),
				iaa.Flipud(0.5, name="Flipud"),
				sometimes(iaa.SomeOf((0, 2), [
						iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
						           rotate=(-25, 25), name="Affine"),
						iaa.ElasticTransformation(alpha=(0.01, 0.1), sigma=0.15,
						                          name="ElasticTransformation"),
						iaa.PiecewiseAffine(scale=(0.001, 0.03), name="PiecewiseAffine"),
						iaa.PerspectiveTransform(scale=(0.01, 0.05), name="PerspectiveTransform"),
				], random_order=True)),

				sometimes(iaa.OneOf([
						iaa.GaussianBlur(sigma=(0, 0.2)),
						iaa.AverageBlur(k=3),
						iaa.MedianBlur(k=3),
				])),

				sometimes(iaa.OneOf([
						iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255)),
						iaa.AddElementwise((-5, 5)),
				])),

				sometimes(iaa.OneOf([
						iaa.GammaContrast(gamma=(0.75, 1.50)),
						iaa.HistogramEqualization(),
						iaa.Multiply((0.80, 1.15)),
						iaa.Add((-20, 15)),
						iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.5)),
						iaa.Emboss(alpha=(0, 0.5), strength=(0.7, 1.5)),
				])),
		], random_order=True)

		seq_det = seq.to_deterministic()
		images = seq_det.augment_images(images)
		targets = seq_det.augment_segmentation_maps([ia.SegmentationMapOnImage(t.astype(bool), shape=t.shape)
			                                             for t in targets])
		targets = np.array([t.get_arr_int() for t in targets])

		return images, targets


def cbam_block(cbam_feature, ratio=2):
	# https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
	#cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=2):
	#channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[-1]

	shared_layer_one = Dense(channel // ratio,
	                         activation='relu',
	                         kernel_initializer='he_normal',
	                         use_bias=True,
	                         bias_initializer='zeros')
	shared_layer_two = Dense(channel,
	                         kernel_initializer='he_normal',
	                         use_bias=True,
	                         bias_initializer='zeros')

	avg_pool = GlobalAveragePooling2D()(input_feature)
	avg_pool = Reshape((1, 1, channel))(avg_pool)
	#assert avg_pool._keras_shape[1:] == (1, 1, channel)
	avg_pool = shared_layer_one(avg_pool)
	#assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
	avg_pool = shared_layer_two(avg_pool)
	#assert avg_pool._keras_shape[1:] == (1, 1, channel)

	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1, 1, channel))(max_pool)
	#assert max_pool._keras_shape[1:] == (1, 1, channel)
	max_pool = shared_layer_one(max_pool)
	#assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
	max_pool = shared_layer_two(max_pool)
	#assert max_pool._keras_shape[1:] == (1, 1, channel)

	cbam_feature = Add()([avg_pool, max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)

	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, kernel_size=7):
	avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
	#assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
	#assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=-1)([avg_pool, max_pool])
	#assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters=1,
	                      kernel_size=kernel_size,
	                      strides=1,
	                      padding='same',
	                      activation='sigmoid',
	                      kernel_initializer='he_normal',
	                      use_bias=False)(concat)
	#assert cbam_feature._keras_shape[-1] == 1
	return multiply([input_feature, cbam_feature])


def unet(pretrained_weights=None, input_size=(128, 128, 1)):
	def dice_coef(y_true, y_pred, smooth=1):
		intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
		return (2. * intersection + smooth) / (
				K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

	def dice_coef_loss(y_true, y_pred):
		return 1 - dice_coef(y_true, y_pred)

	inputs = Input(input_size)
	# encoder
	d1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
	d1 = BatchNormalization()(d1)
	d1 = Conv2D(32, 3, activation='relu', padding='same')(d1)
	d1 = BatchNormalization()(d1)
	d1 = SpatialDropout2D(0.1)(d1)
	d1 = cbam_block(d1)

	d2 = MaxPooling2D(pool_size=(2, 2))(d1)
	d2 = Conv2D(64, 3, activation='relu', padding='same')(d2)
	d2 = BatchNormalization()(d2)
	d2 = Conv2D(64, 3, activation='relu', padding='same')(d2)
	d2 = BatchNormalization()(d2)
	d2 = SpatialDropout2D(0.1)(d2)
	d2 = cbam_block(d2)
	
	d3 = MaxPooling2D(pool_size=(2, 2))(d2)
	d3 = Conv2D(128, 3, activation='relu', padding='same')(d3)
	d3 = BatchNormalization()(d3)
	d3 = Conv2D(128, 3, activation='relu', padding='same')(d3)
	d3 = BatchNormalization()(d3)
	d3 = SpatialDropout2D(0.2)(d3)
	d3 = cbam_block(d3)

	d4 = MaxPooling2D(pool_size=(2, 2))(d3)
	d4 = Conv2D(256, 3, activation='relu', padding='same')(d4)
	d4 = BatchNormalization()(d4)
	d4 = Conv2D(256, 3, activation='relu', padding='same')(d4)
	d4 = BatchNormalization()(d4)
	d4 = SpatialDropout2D(0.3)(d4)
	d4 = cbam_block(d4)
	

	u3 = UpSampling2D(size=(2, 2))(d4)
	u3 = Conv2D(128, 2, activation='relu', padding='same')(u3)
	u3 = BatchNormalization()(u3)
	u3 = concatenate([d3, u3], axis=-1)
	u3 = Conv2D(128, 3, activation='relu', padding='same')(u3)
	u3 = BatchNormalization()(u3)
	u3 = Conv2D(128, 3, activation='relu', padding='same')(u3)
	u3 = BatchNormalization()(u3)
	u3 = cbam_block(u3)

	u2 = UpSampling2D(size=(2, 2))(u3)
	u2 = Conv2D(64, 2, activation='relu', padding='same')(u2)
	u2 = BatchNormalization()(u2)
	u2 = concatenate([d2, u2], axis=-1)
	u2 = Conv2D(64, 3, activation='relu', padding='same')(u2)
	u2 = BatchNormalization()(u2)
	u2 = Conv2D(64, 3, activation='relu', padding='same')(u2)
	u2 = BatchNormalization()(u2)
	u2 = cbam_block(u2)

	u1 = UpSampling2D(size=(2, 2))(u2)
	u1 = Conv2D(32, 2, activation='relu', padding='same')(u1)
	u1 = BatchNormalization()(u1)
	u1 = concatenate([d1, u1], axis=-1)
	u1 = Conv2D(32, 3, activation='relu', padding='same')(u1)
	u1 = BatchNormalization()(u1)
	u1 = Conv2D(32, 3, activation='relu', padding='same')(u1)
	u1 = BatchNormalization()(u1)
	u1 = cbam_block(u1)

	out = Conv2D(3, 3, activation='relu', padding='same')(u1)
	out = BatchNormalization()(out)
	out = cbam_block(out)
	out = Conv2D(1, 1, activation='sigmoid')(out)

	model = Model(inputs=inputs, outputs=out)

	# optimizer = Adam(lr=0.001)
	optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["acc", dice_coef])

	if pretrained_weights is not None:
		model.load_weights(pretrained_weights)

	return model


def normalize_img(img):
	return ((img - img.min()) * (255 / (img - img.min()).max())).astype(np.uint8)


def determine_padding_needed(img_shape, patch_shape):
	px, py = 0, 0
	for px in range(patch_shape[0]):
		if (img_shape[0] + px) % patch_shape[0] == 0:
			break
	for py in range(patch_shape[1]):
		if (img_shape[1] + py) % patch_shape[1] == 0:
			break
	return px, py


def deconstruct_images(image_ids, drive_path, save_path="./PatchDataset/",
                       patch_size=(64,64), image_size=(700, 605),
                       overlap_ratio=2, border=0,
                       save_patches=False):

	os.makedirs(save_path + "images/", exist_ok=True)
	os.makedirs(save_path + "gt/", exist_ok=True)

	px, py = determine_padding_needed(image_size, patch_size)

	pside = (patch_size[0]//2,
	         patch_size[1]//2)

	patches = (np.arange(pside[0], image_size[0]+px-pside[0]+1, patch_size[0]//overlap_ratio - 2*border),
	           np.arange(pside[1], image_size[1]+py-pside[1]+1, patch_size[1]//overlap_ratio - 2*border))

	patch_imgs, patch_segs = [], []
	n_patches = 0
	for img_id in image_ids:
		print("Extracting patches from -> ", img_id)
		img = plt.imread(drive_path + "images/" + img_id + ".ppm")[:,:,1]
		seg = plt.imread(drive_path + "gt/" + img_id + ".ah.ppm")
		img = np.pad(img, ((0, py),(0, px)), mode='constant')
		seg = np.pad(seg, ((0, py),(0, px)), mode='constant')

		for y, ypatch in enumerate(patches[1]):
			for x, xpatch in enumerate(patches[0]):
				n_patches += 1
				patch_img = img[ypatch-pside[0]:ypatch+pside[0], xpatch-pside[1]:xpatch+pside[1]]
				patch_seg = seg[ypatch-pside[0]:ypatch+pside[0], xpatch-pside[1]:xpatch+pside[1]]

				patch_imgs.append(patch_img)
				patch_segs.append(patch_seg)
				if save_patches:
					imageio.imwrite(f"{save_path}images/{img_id}_p{x:02}{y:02}.tif", normalize_img(patch_img))
					imageio.imwrite(f"{save_path}gt/{img_id}_p{x:02}{y:02}.tif", patch_seg)

	print("Total number of patches extracted:", n_patches)
	return patch_imgs, patch_segs


def group_patches(patch_paths):
	images = {}
	pattern = re.compile("im[0-9]{4}.*.tif")
	for p in patch_paths:
		img_id = re.search(pattern, p).group()[:-10]
		if img_id not in images:
			images[img_id] = [p]
		else:
			images[img_id].append(p)
	return images



def reconstruct_image(patch_paths, patches):
	# need to undo overlap and border values
	shape_substring = sorted(patch_paths)[-1][-8:-4]
	img_dims = (int(shape_substring[0:2]),
	            int(shape_substring[2:4]))
	patch_matrix = [[0 for x in range(img_dims[0]+1)] for y in range(img_dims[1]+1)]
	for patch_id, patch in zip(patch_paths, patches):
		pos_substring = patch_id[-8:-4]
		pos = (int(pos_substring[0:2]),
		       int(pos_substring[2:4]))
		patch_matrix[pos[1]][pos[0]] = patch
	img = np.block(patch_matrix)

	px, py = determine_padding_needed(IMAGE_SIZE, patch.shape)  # get patch size somewhere
	img = img[:-py, :-px]

	return img


def split_data_fundus(drive_path, data_split=(0.65, 0.85, 1.0), shuffle=True, num=np.inf):
	original_ids = [t.replace(".ah.ppm", "") for t in os.listdir(drive_path + "gt/")
	                if ('.ah.ppm' in t)]

	if shuffle: np.random.shuffle(original_ids)
	original_ids = original_ids[:min(num, 9999999)]

	split = (int(data_split[0] * len(original_ids)),
	         int(data_split[1] * len(original_ids)),
	         int(data_split[2] * len(original_ids)))

	train_ids = original_ids[0:split[0]]
	val_ids = original_ids[split[0]:split[1]]
	test_ids = original_ids[split[1]:split[2]]
	extra_ids = ["im0291", "im0319", "im0324"]

	return train_ids, val_ids, test_ids


def split_data_drive(drive_path, data_split=(0.8, 1.0), shuffle=True, num=np.inf):
	original_ids = sorted([t.replace(".ah.ppm", "") for t in os.listdir(drive_path + "gt/")
	                if ('.ah.ppm' in t)], key=str.lower)
					
	test_val_ids = original_ids[:20]
	test_ids = original_ids[20:]
	
	if shuffle: np.random.shuffle(test_val_ids)
	split = (int(data_split[0] * len(test_val_ids)),
	         int(data_split[1] * len(test_val_ids)))
	
	train_ids = test_val_ids[0:split[0]]
	val_ids = test_val_ids[split[0]:split[1]]
	
	return train_ids, val_ids, test_ids
	

def load_patch_data(dataset_path, image_ids):
	patches = [dataset_path + "images/" + p
	           for img_id in image_ids
	           for p in os.listdir(dataset_path + "images/")
	           if img_id in p]

	targets = [dataset_path + "gt/" + p
	           for img_id in image_ids
	           for p in os.listdir(dataset_path + "gt/")
	           if img_id in p]

	return patches, targets


def calc_metrics(maskpred, masktrue, mask=None):
	if mask is not None:
		maskpred = np.extract(mask, maskpred)
		masktrue = np.extract(mask, masktrue)
	((TP, FP), (FN, TN)) = metrics.confusion_matrix(masktrue, maskpred)
	return {'Accuracy:   ': round((TP + TN) / (TP + FP + FN + TN), 4),
			'Sensitivity:': round(TP / (TP + FN), 4),
			'Specitivity:': round(TN / (TN + FP), 4),
			'Precision:  ': round(TP / (TP + FP), 4),
			}



def faz_mascaras_classifier(data, drive_path, save_path,
                            classifier, patch_dimensions,
                            present_metrics=True, show_sample_patches=False):
	"""
	Segmenta varias imagens contidas numa diretoria(data_path+database) e guarda-as noutra diretoria(save_path)
	Calcula a media das metricas para o diretorio e apresenta as metricas a cada imagem
	"""

	os.makedirs(save_path, exist_ok=True)

	# metricas gerais da segemntacao da diretoria
	metricas_totais = {'Accuracy:   ': 0,
	                   'Sensitivity:': 0,
	                   'Specitivity:': 0,
	                   'Precision:  ': 0,
	                   }

	start_overal_time = perf_counter()
	for i, (img_id, img_patches_path) in enumerate(group_patches(data).items()):
		start_segment_time = perf_counter()

		img_generator = DataGenerator(images_paths=img_patches_path,
		                              target_paths=None,
		                              image_dimensions=(*patch_dimensions, 1),
		                              batch_size=1, #len(img_patches),
		                              shuffle=False,
		                              augment=False,
		                              )

		segment_patches = classifier.predict_generator(generator=img_generator,
		                                               steps=len(img_generator)
		                                               )

		segment_patches = np.squeeze(segment_patches)
		#segment_patches = [normalize_img(p) for p in segment_patches]

		segment_vasos = normalize_img(reconstruct_image(img_patches_path, segment_patches))

		mask = np.where((plt.imread(drive_path + "masks/" + img_id + "-msf.png") > 0), 255, 0)
		mask = skmorphology.remove_small_holes(mask.astype(bool), area_threshold=5).astype(np.uint8)
		segment_vasos[mask == 0] = 0

		if present_metrics:
			threshold = 128  # skfilters.threshold_otsu(segment_vasos)
			segment_vasos = np.where(segment_vasos > threshold, 255, 0).astype(np.uint8)

			target = np.where((plt.imread(drive_path + "gt/" + img_id + ".ah.ppm") > 0), 255, 0).astype(np.uint8)

			metricas = calc_metrics(segment_vasos, target, mask)

			# acumulacão das metrica de cada imagem para calculo geral posterior
			for m, score in metricas.items():
				metricas_totais[m] += score

			print(f"Image '{img_id}' segmentation results:")
			for m, value in metricas.items():
				print(f"{m}         {value}")
			print(f"Processing time:     {round(perf_counter() - start_segment_time, 5)}s")

			if show_sample_patches:
				# visualiza patches aleatorios
				'''
				idx = np.random.choice(list(range(len(segment_patches))),
				                       size=min(len(segment_patches), 2), replace=False)
				for pi in idx:
					p = segment_patches[pi]
					# print(p.shape, p.min(), p.max(), p.dtype)
					plt.imshow(p, cmap="gray")
					plt.show()
				'''
				# visualiza segmentação
				# print(segment_vasos.min(), segment_vasos.max(), segment_vasos.dtype)
				fig, ax = plt.subplots(1, 2, figsize=(20, 10))
				ax[0].imshow(segment_vasos, cmap="gray")
				ax[0].set_title("Segmentação UNet", fontsize=15)
				ax[1].imshow(target, cmap="gray")
				ax[1].set_title("Target", fontsize=15)
				plt.show()
				

		new_path = save_path + img_id + "_unet.tif"
		imageio.imsave(new_path, segment_vasos)

	if present_metrics:
		# apresenta a medias das metricas associadas a segmentacao das imagens na diretoria
		print("\n\n ----- Validation overall metrics: ----- ")
		for m, score in metricas_totais.items():
			print(m, round(score/(i+1), 5))
		print(f"Total time: {round(perf_counter() - start_overal_time, 5)}s")





def analyse_history(history):
	fig, ax = plt.subplots(3, 1, figsize=(6, 6))
	ax[0].plot(history.history['loss'], label="TrainLoss")
	ax[0].plot(history.history['val_loss'], label="ValLoss")
	ax[0].legend(loc='best', shadow=True)

	ax[1].plot(history.history['dice_coef'], label="TrainDiceCoef")
	ax[1].plot(history.history['val_dice_coef'], label="ValDiceCoef")
	ax[1].legend(loc='best', shadow=True)

	ax[2].plot(history.history['acc'], label="TrainAcc")
	ax[2].plot(history.history['val_acc'], label="ValAcc")
	ax[2].legend(loc='best', shadow=True)
	plt.show()



def delete_old_patches(directory):
	print("Removing old files from: ", directory)
	try:
		images = [os.path.join(directory + "images/", f) for f in os.listdir(directory + "images/")]
		targets = [os.path.join(directory + "gt/", f) for f in os.listdir(directory + "gt/")]
		for f in images + targets:
			os.remove(f)
	except:
		pass






if __name__ == "__main__":
	dataset = "fundus"
	LOAD_MODEL = False
	PATCH_DIMENSIONS = (160, 160)
	EPOCHS = 16
	
	
	#print(os.listdir('../working/'))
	if dataset == "drive":
		drive_path = '../input/drive-like-fundus/drive_like_fundus/DRIVE_like_FUNDUS/'
		IMAGE_SIZE = (565, 584)
		train_ids, val_ids, test_ids = split_data_drive(drive_path=drive_path)
	else:
		drive_path = '../input/fundus20/fundus_original_backup/fundus_original_backup/'
		IMAGE_SIZE = (700, 605)
		train_ids, val_ids, test_ids = split_data_fundus(drive_path=drive_path)
	save_path = '../working/SaveVasosUNet/'
	dataset_path = '../working/PatchDataset/'
    
    
	print("Loading data")
	delete_old_patches(dataset_path)
	

	if not LOAD_MODEL:
		print("Training Model")
		# create train patches
		deconstruct_images(train_ids + val_ids,
		                   drive_path=drive_path,
		                   save_path=dataset_path,
		                   patch_size=PATCH_DIMENSIONS,
		                   image_size=IMAGE_SIZE,
		                   overlap_ratio=18,
		                   border=0,
		                   save_patches=True)

		(Xtrain_paths, Ytrain_paths) = load_patch_data(dataset_path, train_ids)
		(Xval_paths, Yval_paths) = load_patch_data(dataset_path, val_ids)

		train_data = DataGenerator(images_paths=Xtrain_paths,
		                           target_paths=Ytrain_paths,
		                           image_dimensions=(*PATCH_DIMENSIONS, 1),
		                           batch_size=8,
		                           shuffle=True,
		                           augment=True)

		val_data = DataGenerator(images_paths=Xval_paths,
		                         target_paths=Yval_paths,
		                         image_dimensions=(*PATCH_DIMENSIONS, 1),
		                         batch_size=128)

		learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
		                                            patience=2,
		                                            verbose=1,
		                                            factor=0.1,
		                                            min_lr=0.000001)

		checkpoint = ModelCheckpoint(f'new_trained_model_{dataset}.hdf5',
		                             monitor='val_loss',
		                             mode='min',
		                             save_best_only=True,
		                             verbose=1)

		early_stop = EarlyStopping(monitor="val_loss",
		                           mode="min",
		                           patience=5)

		model = unet(pretrained_weights=None,
		             input_size=(*PATCH_DIMENSIONS, 1))
		model.summary()
		
		#class_weights = class_weight.compute_class_weight('balanced',
		#                                                  np.unique(train_data.target_paths),
		#                                                  train_data.target_paths)
		                                                  
		history = model.fit_generator(generator=train_data,
		                              validation_data=val_data,
		                              epochs=EPOCHS,
		                              steps_per_epoch=len(train_data),
		                              #class_weight=class_weights,
		                              callbacks=[learning_rate_reduction, checkpoint, early_stop],
		                              verbose=2)

		analyse_history(history)

		print("Segmenting test images")
		deconstruct_images(test_ids,
		                   patch_size=PATCH_DIMENSIONS,
		                   drive_path=drive_path,
		                   save_path=dataset_path,
		                   image_size=IMAGE_SIZE,
		                   overlap_ratio=1,
		                   border=0,
		                   save_patches=True)

		(Xtest_paths, Ytest_paths) = load_patch_data(dataset_path, test_ids)

		faz_mascaras_classifier(data=Xtest_paths,
		                        drive_path=drive_path,
		                        save_path=save_path,
		                        classifier=model,
		                        patch_dimensions=PATCH_DIMENSIONS,
		                        present_metrics=True,
		                        show_sample_patches=False)




	else:
		print("Loading pre-trained model")
		if dataset == "drive":  weights = '../input/trained-model/trained_model_drive.hdf5'
		else:                   weights = '../input/trained-model-fundus/trained_model_fundus.hdf5'
		model = unet(pretrained_weights = weights,
		             input_size=(*PATCH_DIMENSIONS, 1))
		model.summary()



		print("Segmenting all images")
		deconstruct_images(train_ids + val_ids + test_ids,
		                   patch_size=PATCH_DIMENSIONS,
		                   drive_path=drive_path,
		                   save_path=dataset_path,
		                   image_size=IMAGE_SIZE,
		                   overlap_ratio=1,
		                   border=0,
		                   save_patches=True)

		(Xtest_paths, Ytest_paths) = load_patch_data(dataset_path, train_ids + val_ids + test_ids)

		faz_mascaras_classifier(data=Xtest_paths,
		                        drive_path=drive_path,
		                        save_path=save_path,
		                        classifier=model,
		                        patch_dimensions=PATCH_DIMENSIONS,
		                        present_metrics=True,
		                        show_sample_patches=False)
		                        

	delete_old_patches(dataset_path)