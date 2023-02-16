
import os, pickle
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from vit_pytorch.vivit import ViT
import tqdm
import PIL
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class EmotionDataset(Dataset):
    def __init__(self, root_dir, labels_file, AudioDataset, transform=None):
        self.label_map = {'neutral': 0, 'joy': 1, 'fear': 2, 'anger': 3, 'surprise': 4, 'sadness': 5, 'disgust': 6}
        self.root_dir = root_dir
        self.labels = pd.read_csv(labels_file)
        self.labels['Emotion'] = self.labels['Emotion'].map(self.label_map)
        self.transform = transform
        self.audio = AudioDataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        subfolder_names = os.listdir(self.root_dir)
        video_folder = os.path.join(self.root_dir, subfolder_names[idx])
        video = []
        dia, utt = subfolder_names[idx].split("_")[0][3:], subfolder_names[idx].split("_")[1][3:]
        dia, utt = int(dia), int(utt)
        for i in range(8):
            image_path = os.path.join(video_folder, 'img_%05d.jpg' % i)
            image = PIL.Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            video.append(image)
        video = torch.stack(video, dim=0)
        video = video.permute(1, 0, 2, 3)
        if len(self.labels.query("Dialogue_ID == @dia and Utterance_ID == @utt")['Emotion']) == 0:
            label = 0
            text = "hey"
        else:
            label = int(self.labels.query("Dialogue_ID == @dia and Utterance_ID == @utt")['Emotion'])
            text = self.labels.query("Dialogue_ID == @dia and Utterance_ID == @utt")['Utterance'].values[0]
        audio = torch.FloatTensor(self.audio[f'{dia}_{utt}'][0])
        return video, label, text, audio


class TextEmbedding():
    def __init__(self):
        super(TextEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # pre-trained bert model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # tokenizer

    def embedding(self, text):
        # text: the texts to be tokenized
        # padding: max length: pad the texts to the maximum length (so that all outputs have the same length)
        # return_tensors='pt' : return the tensors (not lists)
        encoding = self.tokenizer(text, padding='max_length', max_length=70, return_tensors='pt')

        with torch.no_grad():
            embedded = self.bert(**encoding)[0]  # put encoded text through bert model

        return embedded


# Hyperparams
max_length = 50  # Maximum length of the sentence


class AudioDataLoader:

    def __init__(self, mode=None):

        self.MODE = 'Emotion'  # Sentiment or Emotion classification mode
        self.max_l = max_length

        """
            Loading the dataset: 
                - revs is a dictionary with keys/value: 
                    - text: original sentence
                    - split: train/val/test :: denotes the which split the tuple belongs to
                    - y: label of the sentence
                    - dialog: ID of the dialog the utterance belongs to
                    - utterance: utterance number of the dialog ID
                    - num_words: number of words in the utterance
                - W: glove embedding matrix
                - vocab: the vocabulary of the dataset
                - word_idx_map: mapping of each word from vocab to its index in W
                - label_index: mapping of each label (emotion or sentiment) to its assigned index, eg. label_index['neutral']=0
        """
        x = pickle.load(open("fusion/data_emotion.p".format(self.MODE.lower()), "rb"))  # CHANGE PATH TO PICKLE FILES
        revs, self.W, self.word_idx_map, self.vocab, _, label_index = x[0], x[1], x[2], x[3], x[4], x[5]
        self.num_classes = len(label_index)
        # print("Labels used for this classification: ", label_index)

        # Preparing data
        self.train_data, self.val_data, self.test_data = {}, {}, {}
        for i in range(len(revs)):

            utterance_id = revs[i]['dialog'] + "_" + revs[i]['utterance']
            sentence_word_indices = self.get_word_indices(revs[i]['text'])
            label = label_index[revs[i]['y']]

            if revs[i]['split'] == "train":
                self.train_data[utterance_id] = (sentence_word_indices, label)
            elif revs[i]['split'] == "val":
                self.val_data[utterance_id] = (sentence_word_indices, label)
            elif revs[i]['split'] == "test":
                self.test_data[utterance_id] = (sentence_word_indices, label)

        # Creating dialogue:[utterance_1, utterance_2, ...] ids
        self.train_dialogue_ids = self.get_dialogue_ids(self.train_data.keys())
        self.val_dialogue_ids = self.get_dialogue_ids(self.val_data.keys())
        self.test_dialogue_ids = self.get_dialogue_ids(self.test_data.keys())

        # Max utternance in a dialog in the dataset
        self.max_utts = self.get_max_utts(self.train_dialogue_ids, self.val_dialogue_ids, self.test_dialogue_ids)

    def get_word_indices(self, data_x):
        length = len(data_x.split())
        return np.array([self.word_idx_map[word] for word in data_x.split()] + [0] * (self.max_l - length))[:self.max_l]

    def get_dialogue_ids(self, keys):
        ids = defaultdict(list)
        for key in keys:
            ids[key.split("_")[0]].append(int(key.split("_")[1]))
        for ID, utts in ids.items():
            ids[ID] = [str(utt) for utt in sorted(utts)]
        return ids

    def get_max_utts(self, train_ids, val_ids, test_ids):
        max_utts_train = max([len(train_ids[vid]) for vid in train_ids.keys()])
        max_utts_val = max([len(val_ids[vid]) for vid in val_ids.keys()])
        max_utts_test = max([len(test_ids[vid]) for vid in test_ids.keys()])
        return np.max([max_utts_train, max_utts_val, max_utts_test])

    def get_one_hot(self, label):
        label_arr = [0] * self.num_classes
        label_arr[label] = 1
        return label_arr[:]

    def get_dialogue_audio_embs(self):
        key = list(self.train_audio_emb.keys())[0]
        pad = [0] * len(self.train_audio_emb[key])

        def get_emb(dialogue_id, audio_emb):
            dialogue_audio = []
            for vid in dialogue_id.keys():
                local_audio = []
                for utt in dialogue_id[vid]:
                    try:
                        local_audio.append(audio_emb[vid + "_" + str(utt)][:])
                    except:
                        print(vid + "_" + str(utt))
                        local_audio.append(pad[:])
                for _ in range(self.max_utts - len(local_audio)):
                    local_audio.append(pad[:])
                dialogue_audio.append(local_audio[:self.max_utts])
            return np.array(dialogue_audio)

        self.train_dialogue_features = get_emb(self.train_dialogue_ids, self.train_audio_emb)
        self.val_dialogue_features = get_emb(self.val_dialogue_ids, self.val_audio_emb)
        self.test_dialogue_features = get_emb(self.test_dialogue_ids, self.test_audio_emb)

    def get_dialogue_labels(self):

        def get_labels(ids, data):
            dialogue_label = []

            for vid, utts in ids.items():
                local_labels = []
                for utt in utts:
                    local_labels.append(self.get_one_hot(data[vid + "_" + str(utt)][1]))
                dialogue_label += local_labels

            return np.array(dialogue_label)

        self.train_dialogue_label = get_labels(self.train_dialogue_ids, self.train_data)

        self.val_dialogue_label = get_labels(self.val_dialogue_ids, self.val_data)

        self.test_dialogue_label = get_labels(self.test_dialogue_ids, self.test_data)

    def get_dialogue_lengths(self):

        self.train_dialogue_length, self.val_dialogue_length, self.test_dialogue_length = [], [], []
        for vid, utts in self.train_dialogue_ids.items():
            self.train_dialogue_length.append(len(utts))
        for vid, utts in self.val_dialogue_ids.items():
            self.val_dialogue_length.append(len(utts))
        for vid, utts in self.test_dialogue_ids.items():
            self.test_dialogue_length.append(len(utts))

    def get_masks(self):

        self.train_mask = np.zeros((len(self.train_dialogue_length), self.max_utts), dtype='float')

        for i in range(len(self.train_dialogue_length)):
            self.train_mask[i, :self.train_dialogue_length[
                i]] = 1.0  # apply importance only to the actual audio lenght, because utterances are padded with zeros

        self.val_mask = np.zeros((len(self.val_dialogue_length), self.max_utts), dtype='float')

        for i in range(len(self.val_dialogue_length)):
            self.val_mask[i, :self.val_dialogue_length[i]] = 1.0

        self.test_mask = np.zeros((len(self.test_dialogue_length), self.max_utts), dtype='float')

        for i in range(len(self.test_dialogue_length)):
            self.test_mask[i, :self.test_dialogue_length[i]] = 1.0

    def load_audio_data(self, ):

        AUDIO_PATH = "fusion/audio_embeddings_feature_selection_emotion.pkl".format(
            self.MODE.lower())  # CHANGE PATH TO PICKLE FILES
        self.train_audio_emb, self.val_audio_emb, self.test_audio_emb = pickle.load(open(AUDIO_PATH, "rb"))

        self.get_dialogue_audio_embs()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()

        for i, key in enumerate(self.train_audio_emb.keys()):  # id, embeddings, and labels
            self.train_audio_emb[key] = [self.train_audio_emb[key], self.train_dialogue_label[i]]

        for i, key in enumerate(self.val_audio_emb.keys()):
            self.val_audio_emb[key] = [self.val_audio_emb[key], self.val_dialogue_label[i]]

        for i, key in enumerate(self.test_audio_emb.keys()):
            self.test_audio_emb[key] = [self.test_audio_emb[key], self.test_dialogue_label[i]]


class FusionLayerMLP(nn.Module):
    '''
    network that is used to transform the unimodal representations into fusion result
    (fusion of video, audio and text modality) for classification
    '''

    def __init__(self, text_dim, video_dim, audio_dim):
        super(FusionLayerMLP, self).__init__()
        self.dim = text_dim + audio_dim + video_dim
        self.batch = 32
        self.textlin = nn.Sequential(
            nn.Linear(53760, 2048),
            nn.Dropout(0.2),  # apply dropout with a probability of 0.2 (or adjust as desired)
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.dim, 2048),  # Insert Input size (text + video + audio)
            nn.Dropout(0.2),
            # output shape of the bert text embedding is [1, 70, 768] for one input (batch size, fixed sentence length, bert output)
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7)  # 7 classes
        )

    def forward(self, text_embedding, audio_embedding, video_embedding):
        text_embedding = text_embedding.to(device)
        audio_embedding = audio_embedding.to(device)
        text_embedding = torch.reshape(text_embedding, (-1, 53760))
        text_embedding = self.textlin(text_embedding)
        fused_embedding = torch.cat((text_embedding, audio_embedding, video_embedding), dim=1)

        pred = self.layers(fused_embedding)  # put through mlp

        return pred


class FusionNetwork(nn.Module):

    def __init__(self, text_dim, video_dim, audio_dim):
        super(FusionNetwork, self).__init__()

        self.v = ViT(
            image_size=256,
            frames=8,
            image_patch_size=16,
            frame_patch_size=2,
            num_classes=2048,
            dim=1024,
            spatial_depth=6,
            temporal_depth=6,
            heads=8,
            mlp_dim=2048,
            channels=1
        )


        self.bert = TextEmbedding()

        self.fusionlayer = FusionLayerMLP(text_dim, video_dim, audio_dim)

    def forward(self, text_data, video_data, audio_data):
        video_embedding = self.v(video_data)
        text_embedding = self.bert.embedding(text_data)

        pred = self.fusionlayer(text_embedding, video_embedding, audio_data)

        return pred


class FusionModelPipeline:
    def __init__(self, model, train_dataloader_video, test_dataloader_video, dev_dataloader_video, device):

        self.model = model
        self.lamda1 = lambda epoch: 0.95

        self.train_dataloader = train_dataloader_video
        self.test_dataloader = test_dataloader_video
        self.dev_dataloader = dev_dataloader_video

        self.device = device
        self.model = self.model.to(self.device)
        self.weights = torch.tensor([1, 3, 15, 4, 3, 6, 15], dtype=torch.float)
        self.weights = self.weights.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, self.lamda1)
        self.scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 50, 60], gamma=0.1)

        self.best_accuracy = 0
        self.num_epochs = 70

    def train(self):
        losses = []
        dev_accuracies = []
        best_dev_accuracy = 0
        for epoch in (pbar := tqdm.tqdm(range(self.num_epochs))):
            correct = 0
            total = 0
            self.model.train()
            for i, (video, labels, text, audio) in enumerate(self.train_dataloader):
                video = video.to(self.device)
                audio = audio.to(self.device)
                #text = text.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(text, video, audio)
                prediction = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += torch.sum(prediction == labels).item()
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 30 == 0:
                    print(f"Processed {i} batches in epoch {epoch}")
                    print(f"prediction: {prediction}")
            if epoch % 5 == 0:
                self.save(f"fusion/ParameterReal{epoch}.pt")
            losses.append(loss.item())
            self.scheduler.step()
            self.scheduler2.step()
            self.model.eval()
            dev_correct = 0
            dev_total = 0
            if epoch % 2 == 0:
                for i, (video, labels, text, audio) in enumerate(self.dev_dataloader):
                    video = video.to(self.device)
                    audio = audio.to(self.device)
                    #text = text.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(text, video, audio)
                    prediction = torch.argmax(outputs.data, 1)
                    dev_total += labels.size(0)
                    dev_correct += torch.sum(prediction == labels).item()
                dev_accuracy = 100 * dev_correct / dev_total
                print(f"Epoch: {epoch} Dev Accuracy: {dev_accuracy}")
                dev_accuracies.append(dev_accuracy)
                df2 = pd.DataFrame({"accuracy": dev_accuracies})
                df2.to_csv("fusion/dev_accuraciesReal.csv", index=False)
                df = pd.DataFrame({"Loss": losses})
                df.to_csv("fusion/lossesReal.csv", index=False)
            accuracy = 100 * correct / total
            print(f"Epoch: {epoch} Accuracy: {accuracy}")



    def test(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (video, labels, text, audio) in enumerate(self.train_dataloader):
                video = video.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(text, video, audio)
                prediction = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += torch.sum(prediction == labels).item()
                print(f"Label: {labels}")
                print(f"prediction: {prediction}")
            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy}")



    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

if __name__ == "__main__":
    text_dim = 2048
    video_dim = 2048
    audio_dim = 1611

    data = AudioDataLoader(mode='audio')
    data.load_audio_data()
    train_data = data.train_audio_emb  # fromat: dictionary with key = ['dialogeid'_'utterance_id'], value = [(audio embedding, label in one-hot format)]
    val_data = data.val_audio_emb
    test_data = data.test_audio_emb

    val_data[f'{49}_{4}'] = val_data[f'{49}_{3}']
    val_data[f'{49}_{5}'] = val_data[f'{49}_{3}']
    val_data[f'{66}_{10}'] = val_data[f'{66}_{5}']
    val_data[f'{66}_{9}'] = val_data[f'{66}_{5}']

    test_data[f'{108}_{1}'] = test_data[f'{108}_{0}']
    test_data[f'{108}_{2}'] = test_data[f'{108}_{0}']
    test_data[f'{93}_{5}'] = test_data[f'{93}_{4}']
    test_data[f'{93}_{6}'] = test_data[f'{93}_{4}']
    test_data[f'{93}_{7}'] = test_data[f'{93}_{4}']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = EmotionDataset(
        'fusion/final_train_splits',
        'fusion/train_sent_emo.csv',
        train_data,
    transform = transform)


    train_dataloader = DataLoader(train_dataset, batch_size=32,
                                  shuffle=True, num_workers=0)

    dev_dataset = EmotionDataset(
        'fusion/final_dev_splits',
        'fusion/dev_sent_emo.csv',
        val_data,
    transform = transform)

    dev_dataloader = DataLoader(dev_dataset, batch_size=32,
                                shuffle=False, num_workers=0)

    test_dataset = EmotionDataset(
        'fusion/final_test_splits',
        'fusion/test_sent_emo.csv',
        test_data,
    transform = transform)

    test_dataloader = DataLoader(test_dataset, batch_size=32,
                                 shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionNetwork(text_dim, video_dim, audio_dim)
    model = FusionModelPipeline(model, train_dataloader, test_dataloader, dev_dataloader, device)
    model.train()



