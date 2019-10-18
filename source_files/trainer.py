import numpy as np
import torch
import time
from torch import nn
from reader import TrainDataReaderWithHSL, TestDataReaderWithHSL
from collections import OrderedDict
from efficient import EfficientNet
import os, datetime, shutil

# TrainTrack Bot imports
from traintrack import TrainTrack
telegram_token = "934547307:AAEb1Pqhk2iXvPrs1pLfHCrZTPehSx0dIkU"  # bot's token
# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = 734383954  # telegram user id (integer):
# Create a TrainTrack Bot instance
bot = TrainTrack(token=telegram_token, user_id=telegram_user_id)
# Activate the bot
bot.activate_bot()

print("START TIME:", datetime.datetime.now())
result_path = "results"
clear_results = True
add_up = False

if clear_results:
    try:
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.mkdir(result_path)
    except OSError:
        print("Error clearing the %s directory" % result_path)
    else:
        print("Successfully cleared previous results in %s directory" % result_path)

# Specify which GPUs to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DIR = "/home/batuhanfaik/DLEpochBot/imagenet_dataset/"

num_classes = 6

# model = torch.load("/home/batuhanfaik/DLEpochBot/source_files/model_saves/Model_85.pt")
model = EfficientNet.from_pretrained('efficientnet-b7')

BATCH_SIZE = 16

train_loader = torch.utils.data.DataLoader(TrainDataReaderWithHSL("train", directory=DIR), batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=64)
test_loader = torch.utils.data.DataLoader(TestDataReaderWithHSL("test", directory=DIR), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=64)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Add fully connected classifier
classifier = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(2560, 1280)),
    ("relu", nn.ReLU()),
    ("drop1", nn.Dropout(0.4)),
    ("fc2", nn.Linear(1280, 640)),
    ("relu", nn.ReLU()),
    ("drop2", nn.Dropout(0.4)),
    ("out_layer", nn.Linear(640, num_classes))
]))

model._fc = classifier
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5, nesterov=True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

model.cuda()
loss = torch.nn.CrossEntropyLoss()

num_epochs = 100

train_loss = []
test_loss = []
total_train_loss = np.empty((num_epochs, 1))
total_train_acc = np.empty((num_epochs, 1))
total_test_loss = np.empty((num_epochs, 1))
total_test_acc = np.empty((num_epochs, 1))

for epoch_id in range(1, num_epochs+1):
    # Bot implementation
    bot.update_epoch(epoch_id)
    if bot.stop_train_flag:
        break
    if bot.lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = bot.lr

    # Don't use a scheduler, manually implement adaptive learning rate
    if epoch_id % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 1.5

    model.train()

    total_loss = 0
    total_true = 0
    total_false = 0
    time_start = time.time()

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = batch[0].cuda()  # input is a 5d tensor, target is 4d
        img_class = batch[1].cuda()
        # bs, ncrops, c, h, w = inputs.size()
        # output = model(inputs.view(-1, c, h, w))  # fuse batch size and ncrops
        # output = output.view(bs, ncrops, -1).mean(1)  # avg over crops
        output = model(inputs)

        _, prediction = torch.max(output.data, 1)
        loss_value = loss(output, img_class)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.data
        total_true += torch.sum(prediction == img_class.data)
        total_false += torch.sum(prediction != img_class.data)

        if (i + 1) % 100 == 0:
            message = "~~Pre-report~~\n" \
                      "Loss: {0:.6f}\n" \
                      "Status: {1} / {2}\n".format(total_loss / len(train_loader), i + 1, len(train_loader)) \
                      + "*" * 16

            print(message)
            # Bot implementation
            bot.prereport_update(message)

    current_train_acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())
    current_train_loss = total_loss.cpu() / len(train_loader)

    message1 = "~~Epoch {0} Report~~\n" \
               "Loss: {1:.6f}\n" \
               "Accuracy: {2:.4f}%\n" \
               "Time (s): {3:.4f}\n".format(epoch_id, total_loss / len(train_loader), current_train_acc * 100,
                                            time.time() - time_start)

    print(message1)
    # Bot implementation
    bot.update_message(message1)

    total_train_loss[epoch_id-1] = current_train_loss
    total_train_acc[epoch_id-1] = current_train_acc

    with torch.no_grad():
        model.eval()

        total_loss = 0
        total_true = 0
        total_false = 0
        time_start = time.time()

        for i, batch in enumerate(test_loader):
            inputs = batch[0].cuda()  # input is a 5d tensor, target is 4d
            img_class = batch[1].cuda()
            bs, ncrops, c, h, w = inputs.size()
            output = model(inputs.view(-1, c, h, w))  # fuse batch size and ncrops
            output = output.view(bs, ncrops, -1).mean(1)  # avg over crops
            # output = model(inputs)

            _, prediction = torch.max(output.data, 1)

            total_loss += loss(output, img_class).data
            total_true += torch.sum(prediction == img_class.data)
            total_false += torch.sum(prediction != img_class.data)

    # scheduler.step(total_loss)
    current_test_acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())

    message2 = "~~Validation {0} Report~~\n" \
                   "Loss: {1:.6f}\n" \
                   "Accuracy: {2:.4f}%\n" \
                   "Time (s): {3:.4f}\n".format(epoch_id, total_loss / len(test_loader), current_test_acc * 100,
                                                time.time() - time_start)

    print(message2)
    # Bot implementation
    bot.update_message(message2)
    status = message1 + "\n" + message2
    bot.set_status(status)

    if epoch_id % 5 == 0:
        model = model.cpu()
        if not os.path.isdir("./model_saves/"):
            os.makedirs("./model_saves/")
        torch.save(model, "./model_saves/Model_%d.pt" % epoch_id)
        model.cuda()

    current_test_loss = total_loss.cpu() / len(test_loader)
    total_test_loss[epoch_id-1] = current_test_loss
    total_test_acc[epoch_id-1] = current_test_acc

    if add_up and os.listdir("./results/"):
        try:
            total_train_loss_saved = np.load("./results/TR_Losses.npy")
            total_test_loss_saved = np.load("./results/TS_Losses.npy")
            total_train_acc_saved = np.load("./results/TR_Acc.npy")
            total_test_acc_saved = np.load("./results/TS_Acc.npy")
            np.save("./results/TR_Losses.npy", np.concatenate((total_train_loss_saved, total_train_loss), axis=0))
            np.save("./results/TS_Losses.npy", np.concatenate((total_test_loss_saved, total_test_loss), axis=0))
            np.save("./results/TR_Acc.npy", np.concatenate((total_train_acc_saved, total_train_acc), axis=0))
            np.save("./results/TS_Acc.npy", np.concatenate((total_test_acc_saved, total_test_acc), axis=0))
        except FileNotFoundError:
            np.save("./results/TR_Losses.npy", total_train_loss)
            np.save("./results/TS_Losses.npy", total_test_loss)
            np.save("./results/TR_Acc.npy", total_train_acc)
            np.save("./results/TS_Acc.npy", total_test_acc)
    else:
        np.save("./results/TR_Losses.npy", total_train_loss)
        np.save("./results/TS_Losses.npy", total_test_loss)
        np.save("./results/TR_Acc.npy", total_train_acc)
        np.save("./results/TS_Acc.npy", total_test_acc)

# Bot implementation
    bot.cumulate_train_loss(current_train_loss)
    bot.cumulate_test_loss(current_test_loss)
    bot.cumulate_train_acc(current_train_acc)
    bot.cumulate_test_acc(current_test_acc)

if bot.stop_train_flag:
    print("Training stopped by {}!".format(bot.name))
    bot.send_message("Training stopped by {}!".format(bot.name))
else:
    print("Training complete. {} out!".format(bot.name))
    bot.send_message("Training complete. {} out!".format(bot.name))

bot.stop_bot()
