BATCHSIZE = 256
BASE_LR = 1e-5
LR = BASE_LR * BATCHSIZE

NUMWORKER = 2  # Colab Free chỉ nên để 2 hoặc ít hơn

CHANNEL_SIZE = 128
EMBEDING_SIZE = 192

NUM_EVAL = 2
EVAL_FRAMES = 300
MAX_FRAMES = 300
SAMPLING_RATE = 16000

TRAIN_DATASET = 'train_dataset_classification'
TRAIN_DATASET_CONFIG = {
  'train_list': '/content/train_list2.txt',
  'train_path': '/root/.cache/kagglehub/datasets/davidthomastran/vietnam-celeb-dataset/versions/1/full-dataset/data',
  'max_frames': MAX_FRAMES,
  'augment': False,
  'musan_path': './data/musan',
  'rir_path': './data/RIRS_NOISES',
}

TEST_LIST = '/content/test_list_space.txt'
TEST_LIST_E = '/content/test_list_e_space.txt'
TEST_LIST_H = '/content/test_list_space.txt'
TEST_DATASET = 'test_dataset'
TEST_DATASET_CONFIG = {
  'test_list': TEST_LIST,
  'test_path': '/root/.cache/kagglehub/datasets/davidthomastran/vietnam-celeb-dataset/versions/1/full-dataset/data',
}

FEATURE_EXTRACTOR = 'mel_transform'
FEATURE_EXTRACTOR_CONFIG = {
  'sample_rate': 16000,
  'n_fft': 512,
  'win_length': 400,
  'hop_length': 160,
  'n_mels': 80,
  'coef': 0.97,
}

SPEC_AUG = 'spec_aug'
SPEC_AUG_CONFIG = {
  'freq_mask_param': 8,
  'time_mask_param': 10,
}

MODEL = 'NeXt_TDNN'
MODEL_CONFIG = {
  'depths': [3, 3, 3],
  'dims': [CHANNEL_SIZE, CHANNEL_SIZE, CHANNEL_SIZE],
  'kernel_size': [7, 65],
  'block': 'TSConvNeXt',
}

AGGREGATION = 'vap_bn_tanh_fc_bn'
AGGREGATION_CONFIG = {
  'channel_size': int(3 * CHANNEL_SIZE),
  'intermediate_size': int(3 * CHANNEL_SIZE / 8),
  'embeding_size': EMBEDING_SIZE,
}

LOSS = 'aamsoftmax'
LOSS_CONFIG = {
  'embeding_size': EMBEDING_SIZE,
  'num_classes': 1000,  # ⚠️ Bạn cần sửa số này đúng với số speaker thật trong train_list.txt
  'margin': 0.3,
  'scale': 40,
}

OPTIMIZER = 'adam'
OPTIMIZER_CONFIG = {
  'lr': LR,
  'weight_decay': 0.01,
}

SCHEDULER = 'steplr'
SCHEDULER_CONFIG = {
  'step_size': 10,
  'gamma': 0.8,
}
#####
ENGINE_CONFIG = {
  'eval_config': {
    'method': 'num_seg',
    'test_list': TEST_LIST,
    'num_eval': NUM_EVAL,
    'eval_frames': EVAL_FRAMES,
    'c_miss': 1,
    'p_target': 0.05,
    'c_fa': 1,
  }
}

CHECKPOINT_CONFIG = {
  'save_top_k': 5,
  'monitor': 'min_eer',
  'mode': 'min',
  'filename': 'colab-{epoch}-{loss:.2f}-{min_eer:.2f}',
}

TRAINER_CONFIG = {
  'default_root_dir': './experiments/NeXt_TDNN_Colab',
  'val_check_interval': 1.0,
  'max_epochs': 123,
  'accelerator': 'gpu',
  'devices': 1,
  'num_sanity_val_steps': 0  # ⚠️ Tắt sanity check trên Colab
}

RESUME_CHECKPOINT = '/content/colab-epoch=116-loss=0.72-min_eer=8.15.ckpt'
PRETRAINED_CHECKPOINT = '/content/colab-epoch=116-loss=0.72-min_eer=8.15.ckpt'
RESTORE_LOSS_FUNCTION = True

TEST_CHECKPOINT = '/content/colab-epoch=116-loss=0.72-min_eer=8.15.ckpt'
TEST_RESULT_PATH = f"{TRAINER_CONFIG.get('default_root_dir')}_test"
TOP_K = 100
COHORT_LIST_PATH = '/content/cohort.csv'
TEST_DATASET_LIST = []
COHORT_SAVE_PATH = './embedding'