from os import listdir
from os.path import join

import cv2 as cv
import mnist
import numpy as np
import torch
import torch.nn.functional as F

from data import SequencePredictionDataset, SequenceReconstructionDataset
from model import CNNPreProcessor, AutoEncoder, VariationalAutoEncoder, SequencePredictor, DeepVariationalBayesFilter
from utils import load_checkpoint, opencv_show, put_side_by_side, NoiseType, add_noise, load_json


def vae_generation():
    target_shape = (64, 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "useful_models/vae_5_layers_mnist_max_lr_0.01_24122020_115212_050.tar"
    # model_path = "useful_models/vae_5_layers_mnist_max_lr_0.01_24122020_141519_050.tar"
    model_path = "useful_models/squeeze_vae_mountain_car_v0_max_lr_0.005_29122020_143138_050.tar"
    input_shape = (1,) + target_shape

    # model = AutoEncoder.get_basic_ae(input_shape=input_shape).to(device)
    # model = VariationalAutoEncoder.get_basic_vae(input_shape=input_shape).to(device)
    model = VariationalAutoEncoder.get_squeeze_vae(input_shape=input_shape).to(device)
    load_checkpoint(model_path, model)

    mnist_preprocessor = CNNPreProcessor(bgr_mean=0.1307, bgr_std=0.3081, target_shape=target_shape)
    mountain_car_64g_preprocessor = CNNPreProcessor(bgr_mean=0.9857, bgr_std=0.1056)
    preprocessor = mountain_car_64g_preprocessor

    for _ in range(100):
        generated = model.generate(device)
        im_generated = preprocessor.reverse_preprocess(generated)

        opencv_show(im_generated)

    print("")


def evaluation():
    # cart_pole_v0_bgr_mean = (0.9890, 0.9898, 0.9908)
    # cart_pole_v0_bgr_std = (0.0977, 0.0936, 0.0906)
    # target_shape = (64, 64)

    cart_pole_v0_bgr_mean = (0.9922, 0.9931, 0.9940)
    cart_pole_v0_bgr_std = (0.0791, 0.0741, 0.0703)
    target_shape = (64, 64)
    latent_dim = 512

    preprocessor = CNNPreProcessor(bgr_mean=cart_pole_v0_bgr_mean, bgr_std=cart_pole_v0_bgr_std,
                                   target_shape=target_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "model_checkpoints/cartpoleV0_autoencoder_2_200.tar"
    # model_path = "model_checkpoints/cartpoleV0_autoencoder_3_032.tar"
    # model_path = "useful_models/cartpoleV0_basic_autoencoder_mnist_14122020_026.tar"
    # model_path = "useful_models/no_out_activation_autoencoder_5_layers_mnist_max_lr_0.01_23122020_160428_049.tar"
    # model_path = "useful_models/squeeze_autoencoder_mnist_max_lr_0.001_26122020_122631_050.tar"
    model_path = "useful_models/squeeze_vae_mnist_max_lr_0.001_26122020_142330_050.tar"

    # model_path = "useful_models/vae_5_layers_mnist_max_lr_0.01_24122020_115212_050.tar"

    input_shape = (1,) + target_shape
    # model = AutoEncoder.get_basic_ae(input_shape=input_shape).to(device)
    # model = AutoEncoder.get_squeeze_ae(input_shape=input_shape).to(device)
    # model = VariationalAutoEncoder.get_basic_vae(input_shape=input_shape).to(device)
    model = VariationalAutoEncoder.get_squeeze_vae(input_shape=input_shape).to(device)
    load_checkpoint(model_path, model)

    ims_path = "agent_frames/cartpoleV0"
    f_names = [join(ims_path, e) for e in listdir(ims_path) if e.endswith(".jpg")]

    mnist_test_ims = [el for el in mnist.test_images()]
    mnist_preprocessor = CNNPreProcessor(bgr_mean=0.1307, bgr_std=0.3081, target_shape=target_shape)
    preprocessor = mnist_preprocessor

    for _ in range(100):
        # random_im_path = np.random.choice(f_names)
        #
        # im = cv.imread(random_im_path)
        im = mnist_test_ims[np.random.randint(len(mnist_test_ims))]

        im_target = cv.resize(im, target_shape)
        orig_shape = im.shape[:2][::-1]
        in_t = preprocessor.preprocess(im).to(device)

        # out_t, embedding = model(in_t)
        out_t, mu, log_var = model(in_t)

        loss = F.mse_loss(in_t, out_t)
        out_im_target = preprocessor.reverse_preprocess(out_t)
        out_im = cv.resize(out_im_target, orig_shape)

        sbs = put_side_by_side([im_target, out_im_target])
        print(f"Loss: {loss.item()}")
        opencv_show(sbs)

    print("")
    pass


def sequence_evaluation():
    mountain_car_v0_64g_mean = 0.9857
    mountain_car_v0_64g_std = 0.1056

    target_shape = (64, 64)

    batch_size = 32
    latent_dim = 128
    num_actions = 3
    seq_length = 30

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    preprocessor = CNNPreProcessor(bgr_mean=mountain_car_v0_64g_mean, bgr_std=mountain_car_v0_64g_std,
                                   target_shape=target_shape, to_grayscale=True)

    data_files_path = "keyboard_agent_frames/MountainCar-v0"
    train_files = load_json("mountain_car_v0_train_sequences_31122020.json")
    val_files = load_json("mountain_car_v0_val_sequences_31122020.json")

    val_dataset = SequencePredictionDataset([data_files_path], batch_size, preprocessor, include_ids=val_files,
                                            prepare_on_load=True, num_actions=num_actions, seq_length=seq_length)

    input_shape = (1,) + target_shape

    rnn_layers = 1
    action_dim = num_actions - 1
    bidirectional = True

    # model_path = "useful_models/squeeze_seq_predictor_mountain_car_v0_max_lr_0.001_03012021_160811_050.tar"
    model_path = "useful_models/squeeze_seq_predictor_seq_length_30_bidirectional_mountain_car_v0_max_lr_0.001_04012021_162444_050.tar"
    model = SequencePredictor.get_squeeze_seq_predictor(input_shape=input_shape, latent_dim=latent_dim,
                                                        action_dim=action_dim, rnn_layers=rnn_layers,
                                                        bidirectional_rnn=bidirectional).to(device)
    load_checkpoint(model_path, model, device=device)
    simulation_steps = 2000
    num_repeated_actions = 100

    model.eval()
    for _ in range(100):
        print("New Sequence")
        random_sequence = val_dataset.sequences[np.random.randint(len(val_dataset.sequences))]

        in_seq = random_sequence[:-1]
        in_im_seq = [el["im"] for el in in_seq]
        in_a_seq = [el["action"] for el in in_seq]
        show_size = (300, 200)
        curr_in_im_seq = in_im_seq
        curr_a_seq = in_a_seq
        curr_a = None

        print(f"Action sequence: {in_a_seq}")
        for sim_step_idx in range(simulation_steps):
            x_t = preprocessor.preprocess_im_sequence(curr_in_im_seq).to(device)
            a_t = val_dataset.actions_to_tensor(curr_a_seq).to(device)

            with torch.no_grad():
                out_t = model(x_t, a_t)
            out_im = preprocessor.reverse_preprocess(out_t, show_size)
            in_seq_sbs = put_side_by_side([cv.resize(el, show_size) for el in curr_in_im_seq])
            if sim_step_idx == 0:
                opencv_show(in_seq_sbs, out_im)
            else:
                cv.imshow("Frame", out_im)
                cv.waitKey(1)

            if sim_step_idx % num_repeated_actions == 0:
                curr_a = np.random.randint(num_actions)
                print(f"Action {curr_a}")

            curr_in_im_seq = curr_in_im_seq[1:] + [out_im]
            curr_a_seq = curr_a_seq[1:] + [curr_a]

    cv.destroyAllWindows()

    print("")

    pass


def dvbf_simulation():
    mountain_car_v0_64g_mean = 0.9857
    mountain_car_v0_64g_std = 0.1056

    target_shape = (64, 64)

    batch_size = 32
    latent_dim = 128
    num_actions = 3

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    preprocessor = CNNPreProcessor(bgr_mean=mountain_car_v0_64g_mean, bgr_std=mountain_car_v0_64g_std,
                                   target_shape=target_shape, to_grayscale=True)

    data_files_path = "keyboard_agent_frames/MountainCar-v0"
    train_files = load_json("mountain_car_v0_train_sequences_31122020.json")
    # val_files = load_json("mountain_car_v0_val_sequences_31122020.json")
    val_files = load_json("mountain_car_v0_big_val_sequences_17012021.json")

    seq_length = 60
    val_dataset = SequenceReconstructionDataset([data_files_path], batch_size, preprocessor, include_ids=val_files,
                                                prepare_on_load=True, num_actions=num_actions, seq_length=seq_length)

    input_shape = (1,) + target_shape

    rnn_layers = 1
    action_dim = num_actions - 1
    bidirectional_rnn = True

    # model_path = "useful_models/squeeze_dvbf_seq_length_30_bidirectional_mountain_car_v0_max_lr_0.0005_11012021_073140_050.tar"
    # model_path = "useful_models/squeeze_dvbf_pretrained_seq_length_100_bidirectional_mountain_car_v0_max_lr_0.0005_14012021_164801_009.tar"
    # model_path = "useful_models/squeeze_dvbf_pretrained_seq_length_60_bidirectional_mountain_car_v0_max_lr_0.0005_14012021_142801_002.tar"
    model_path = "useful_models/squeeze_dvbf_pretrained_seq_length_60_bidirectional_mountain_car_v0_big_max_lr_0.0001_19012021_101834_044.tar"

    model = DeepVariationalBayesFilter.get_squeeze_dvbf_ll(input_shape=input_shape, latent_dim=latent_dim,
                                                           action_dim=action_dim, rnn_layers=rnn_layers,
                                                           bidirectional_rnn=bidirectional_rnn).to(device)
    load_checkpoint(model_path, model, device=device)
    model.eval()

    simulation_steps = 5000
    num_repeated_actions = 60

    for _ in range(100):
        print("New Sequence")
        random_sequence = val_dataset.sequences[np.random.randint(len(val_dataset.sequences))]

        in_seq = random_sequence
        in_im_seq = [el["im"] for el in in_seq]
        in_a_seq = [el["action"] for el in in_seq]
        show_size = (300, 200)
        curr_in_im_seq = in_im_seq
        curr_a_seq = in_a_seq
        curr_a = None

        # decoder_outs, z_s, w_mus, w_log_vars = [], [], [], []

        print(f"Action sequence: {in_a_seq}")
        for sim_step_idx in range(simulation_steps):

            if sim_step_idx == 0:
                x_t = preprocessor.preprocess_im_sequence(curr_in_im_seq).to(device)
                a_t = val_dataset.actions_to_tensor(curr_a_seq).to(device)

                with torch.no_grad():
                    decoder_outs, z_s, w_s, w_mus, w_log_vars = model(x_t, a_t)

                out_ims = [preprocessor.reverse_preprocess(out_t, show_size) for out_t in decoder_outs]
                out_sbs = put_side_by_side(out_ims)
                in_seq_sbs = put_side_by_side([cv.resize(el, show_size) for el in curr_in_im_seq])
                opencv_show(in_seq_sbs, out_sbs, titles=["in_sequence", "out_sequence"])
            else:
                curr_a_t = val_dataset.actions_to_tensor([curr_a]).to(device)
                prev_w = w_s[-1]

                with torch.no_grad():
                    out_t, next_z = model.simulate_next(z_s[-1], curr_a_t[0], device, w=prev_w)
                    out_im = preprocessor.reverse_preprocess(out_t, show_size)
                    cv.imshow("Frame", out_im)
                    cv.waitKey(1)

                    back_encoded_flat = model.encoder(out_t).view(1, -1)
                    next_w, next_w_mu, next_w_log_var = model.get_w(back_encoded_flat)
                    z_s.append(next_z)
                    w_s.append(next_w)
                    w_mus.append(next_w_mu)
                    w_log_vars.append(next_w_log_var)

            if sim_step_idx % num_repeated_actions == 0:
                # curr_a = np.random.randint(num_actions)
                # curr_a = 2 if curr_a == 0 else 0
                curr_a = 1
                print(f"Action {curr_a}")

            curr_a_seq = curr_a_seq[1:] + [curr_a]

    cv.destroyAllWindows()

    print("")


if __name__ == "__main__":
    # evaluation()
    # sequence_evaluation()
    # vae_generation()
    dvbf_simulation()
