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
    -   [Built With](#built-with)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
    -   [implementation](#implementation)
-   [Usage](#usage)
-   [Roadmap](#roadmap)
-   [Contributing](#contributing)
-   [License](#license)
-   [Contact](#contact)
-   [Acknowledgements](#acknowledgements)

<!-- ABOUT THE PROJECT -->

## About The Project

[TrainTrack](https://github.com/batuhanfaik/TrainTrack/) is a [Telegram](https://telegram.org/) bot that helps you track, visualize and control your neural network training process on various platforms using [Telegram Bot
API](https://github.com/python-telegram-bot/python-telegram-bot). It is developed for [PyTorch](https://pytorch.org/) framework.

Origin of the bot and it's implementation on TenserFlow, Keras can be found on [@eyalzk](https://eyalzk.github.io/)'s [GitHub repo](https://github.com/eyalzk/telegrad).

<p align="center">
  <a href="https://github.com/batuhanfaik/TrainTrack">
    <img src="/img/screenshot.png" alt="TrainTrack Screenshot" height="480" />
  </a>
</p>

### Features

-   Get updates on loss, accuracy with personalized messages
-   Control the learning rate
-   Query the latest LR or metrics
-   Get loss, accuracy convergence plots
-   Stop the training process
-   Limit access to a specific Telegram user ID

### Built With

-   [Python](https://www.python.org/)

<!-- GETTING STARTED -->

## Getting Started

Follow these guidelines to get your bot up and running.

### Prerequisites

-   [python-telegram-bot](https://pypi.org/project/python-telegram-bot/)
Required for Telegram bot to function
-   [matplotlib](https://matplotlib.org/users/installing.html)
Required for sending plots to the user
-   [torch](https://pytorch.org/get-started/locally/)
-   [torchvision](https://pypi.org/project/torchvision/)
Required if you want to run pytorch_mnist_example.py

```sh
pip install python-telegram-bot
pip install matplotlib
pip install torch torchvision
```
