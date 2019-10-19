<!-- INTRO -->

<h1 align="center">TrainTrack</h1>
<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack">
    <img src="/img/logo.png" alt="TrainTrack" height="80" />
  </a>
</p>
<p align="center">
  A Telegram bot for controlling NN training process remotely!
  <br/>
  <a href="https://github.com/batuhanfaik/TrainTrack/issues">Report Bug</a>
  ·
  <a href="https://github.com/batuhanfaik/TrainTrack/issues">Request Feature</a>
  ·
  <a href="mailto:batuhan@derinbay.com">Contact Me</a>
</p>
<br/>
<div class="badges_quality">
  <p align="center">
    <a href="https://www.codefactor.io/repository/github/batuhanfaik/traintrack"><img src="https://www.codefactor.io/repository/github/batuhanfaik/traintrack/badge?s=f80b4d54be8cacf5cdd84ed3b022758feef5804c" alt="CodeFactor" /></a>
    <a href="https://scrutinizer-ci.com/g/batuhanfaik/TrainTrack/"><img src="https://scrutinizer-ci.com/g/batuhanfaik/TrainTrack/badges/quality-score.png?b=master&s=c04ea8230d0eceb057e7ce84cf663557fc4c23f5" alt="Scrutinizer Code Quality" /></a>
    <a href="https://scrutinizer-ci.com/g/batuhanfaik/TrainTrack/build-status/master"><img src="https://scrutinizer-ci.com/g/batuhanfaik/TrainTrack/badges/build.png?b=master&s=96213f9f503a3d89c233b2aa976b9c169dcf8f29" alt="Build Status" /></a>
    <a href="https://scrutinizer-ci.com/code-intelligence"><img src="https://scrutinizer-ci.com/g/batuhanfaik/TrainTrack/badges/code-intelligence.svg?b=master&s=b0af3ba8b6ed4f0373456794e98ea534912645eb" alt="Code Intelligence Status" /></a>
  </p>
</div>
<div class="badges_repo">
  <p align="center">
    <a href="https://github.com/batuhanfaik/TrainTrack/graphs/contributors"><img src="https://img.shields.io/github/contributors/batuhanfaik/TrainTrack.svg?style=flat" alt="Contributors" /></a>
    <a href="https://github.com/batuhanfaik/TrainTrack/network/members"><img src="https://img.shields.io/github/forks/batuhanfaik/TrainTrack.svg?style=flat" alt="Forks" /></a>
    <a href="https://github.com/batuhanfaik/TrainTrack/stargazers"><img src="https://img.shields.io/github/stars/batuhanfaik/TrainTrack.svg?style=flat" alt="Stargazers" /></a>
    <a href="https://github.com/batuhanfaik/TrainTrack/issues"><img src="https://img.shields.io/github/issues/batuhanfaik/TrainTrack.svg?style=flat" alt="Issues" /></a>
    <a href="https://github.com/batuhanfaik/TrainTrack/blob/master/LICENSE"><img src="https://img.shields.io/github/license/batuhanfaik/TrainTrack.svg?style=flat" alt="MIT License" /></a>
    <a href="https://linkedin.com/in/batuhanderinbay"><img src="https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555" alt="LinkedIn" /></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->

## Table of Contents

-   [About the Project](#about-the-project)
    -   [Features](#features)
    -   [Built with](#built-with)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
    -   [Implementation](#implementation)
-   [Usage](#usage)
-   [Roadmap](#roadmap)
-   [Contributing](#contributing)
-   [License](#license)
-   [Contact](#contact)
-   [Acknowledgements](#acknowledgements)

<!-- ABOUT THE PROJECT -->

## About the Project

[TrainTrack](https://github.com/batuhanfaik/TrainTrack/) is a [Telegram](https://telegram.org/) bot that helps you track, visualize and control your neural network training process on various platforms using [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) wrapper for Telegram Bot API. TrainTrack is developed for [PyTorch](https://pytorch.org/) framework but can be used with TensorFlow and Keras as well.

Origin of the bot and it's implementation on TenserFlow, Keras can be found on [@eyalzk](https://eyalzk.github.io/)'s [GitHub repo](https://github.com/eyalzk/telegrad).

An example implementation of the bot on [PyTorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist) is in the source files under the filename `pytorch_mnist_example.py`. Read the source code thoroughly for better understanding its implementation on your neural network trainer. The example is well commented and points out each line of code you should implement in order for TrainTrack to function properly.

<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack">
    <img src="/img/screenshot.png" alt="TrainTrack Screenshot" height="240" />
  </a>
</p>

### Features

-   Get updates on loss, accuracy with personalized messages
<details><summary><b>Show message commands</b></summary>
<br/>
<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack#features">
    <img src="/img/update_messages.gif" alt="TrainTrack Update Messages" height="598" />
  </a>
</p>
</details>

-   Control the learning rate
<details><summary><b>Show learning rate commands</b></summary>
<br/>
<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack#features">
    <img src="/img/learning_rate.gif" alt="TrainTrack Learning Rate" height="598" />
  </a>
</p>
</details>

-   Query the latest metrics
<details><summary><b>Show status command</b></summary>
<br/>
<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack#features">
    <img src="/img/status.gif" alt="TrainTrack Status Report" height="598" />
  </a>
</p>
</details>

-   Get loss, accuracy convergence plots
<details><summary><b>Show plot command</b></summary>
<br/>
<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack#features">
    <img src="/img/plot.gif" alt="TrainTrack Plot" height="598" />
  </a>
</p>
</details>

-   Stop the training process
<details><summary><b>Show stop command</b></summary>
<br/>
<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack#features">
    <img src="/img/stop.gif" alt="TrainTrack Stop" height="598" />
  </a>
</p>
</details>

-   Limit access to a specific Telegram user ID

### Built with

-   [Python](https://www.python.org/)
<a href="https://www.python.org/"><img src="https://img.shields.io/github/pipenv/locked/python-version/batuhanfaik/TrainTrack" alt="Python" /></a>

<!-- GETTING STARTED -->

## Getting Started

Follow these guidelines to get TrainTrack up and running!

### Prerequisites

-   [python-telegram-bot](https://pypi.org/project/python-telegram-bot/) | Required for Telegram bot to function
-   [matplotlib](https://matplotlib.org/users/installing.html) | Required for sending plots to the user
-   [torch](https://pytorch.org/get-started/locally/), [torchvision](https://pypi.org/project/torchvision/) | Required if you want to run pytorch_mnist_example.py

```sh
pip install python-telegram-bot
pip install matplotlib
pip install torch torchvision
```

### Installation

1. Make sure you have installed required prerequisites

2. Clone this repository

```sh
git clone https://github.com/batuhanfaik/TrainTrack.git
```

3. Copy `traintrack.py` to your trainer's source file destination

```sh
cd TrainTrack/
cp traintrack.py /<your>/<training_dir>/<source_files>/
```

4. Create a free Telegram bot using the Telegram app. Just follow [these](https://core.telegram.org/bots#6-botfather) instructions provided in the official Telegram bot documentation.

5. Take a note of the given token. It will be used later in the [Implementation](#implementation) section.

  * Optional: You can follow [this](https://bigone.zendesk.com/hc/en-us/articles/360008014894-How-to-get-the-Telegram-user-ID-) article to learn your Telegram user ID. Your user ID can be used to limit the
  access of TrainTrack only to yourself

### Implementation

-  Import bot in your trainer source file

<details><summary><b>Show code</b></summary>

Following piece of code is all you need to import TrainTrack into your project

```python
# Import TrainTrack Bot
from traintrack import TrainTrack

telegram_token = "TOKEN"  # TrainTrack's token
# User id is optional and can be kept as None.
# However highly recommended as it limits the access to you alone.
telegram_user_id = None  # Telegram user id (integer):
# Create a TrainTrack Bot instance
TrainTrack = TrainTrack(token=telegram_token, user_id=telegram_user_id)
# Activate the bot
TrainTrack.activate_bot()
```

</details>

-   Add TrainTrack's `update_epoch()` method, `/stop` and learning rate controlling
commands in the training loop

<details><summary><b>Show code</b></summary>

Following piece of code is needed to be placed in your training loop

```python
# Update the epoch variable in TrainTrack in order to keep track of
# the current epoch
TrainTrack.update_epoch(epoch)
# Force break epoch loop when the user stops training
if TrainTrack.stop_train_flag:
    break
# Manually control learning rate using TrainTrack
if TrainTrack.learning_rate is not None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = TrainTrack.learning_rate
```

</details>

-   Using TrainTrack's `update_prereport()`, `update_message()`, `add_status()`, `clr_status()` methods, you can control the messages that will be sent out by TrainTrack

-   Using TrainTrack's `cumulate_<train/test>_<loss/acc>()` methods, you can cumulate your losses and
accuracies for the plot

-   Add TrainTrack's `stop_bot()` method and exit condition handlers after the training
loop

<details><summary><b>Show code</b></summary>

Following piece of code is needed to be placed after your training loop

```python
# Exit conditions handling for TrainTrack
# Notifies the user whether the training has terminated or finished after completing all epochs
if TrainTrack.stop_train_flag:
    print("Training stopped by {}!".format(TrainTrack.name))
    TrainTrack.send_message("Training stopped by {}!".format(TrainTrack.name))
else:
    print("Training complete. {} out!".format(TrainTrack.name))
    TrainTrack.send_message("Training complete. {} out!".format(TrainTrack.name))
# Stop TrainTrack Bot instance at the end of training
TrainTrack.stop_bot()
```

</details>

<!-- USAGE EXAMPLES -->

## Usage

Each command is shown in the [Features](#features) section. Please refer to it since there is no documentation available for this project yet.

<!-- ROADMAP -->

## Roadmap

See the [open issues](https://github.com/github_username/repo/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/new_feature`)
3.  Commit your Changes (`git commit -m 'Added this new_feature'`)
4.  Push to the Branch (`git push origin feature/new_feature`)
5.  Open a Pull Request

Keep in mind that pull requests are always welcome!

<!-- LICENSE -->

## Licenses

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->

## Contact

<h3 align="center">Batuhan Faik Derinbay</h3>
<p align="center">
  <a href="https://www.linkedin.com/in/batuhanderinbay/">LinkedIn</a>
   |
  <a href="mailto:batuhan@derinbay.com">Email</a>
   |
  <a href="https://github.com/batuhanfaik/traintrack">This Project</a>
</p>

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

-   [telegrad](https://github.com/eyalzk/telegrad)
-   [python-telegram-bot]](https://github.com/python-telegram-bot/python-telegram-bot)
