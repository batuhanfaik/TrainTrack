from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove, Update)
from telegram.ext import (Updater, CommandHandler, Filters, ConversationHandler,
                          MessageHandler, CallbackContext)

import logging
from io import BytesIO

try:
    import matplotlib

    matplotlib.use('Agg')  # Use Agg backend because the QObject is created outside main func
    from matplotlib import pyplot as plt
except ImportError:
    plt = None


class TrainTrack(object):
    """  A class for interacting with a Telegram bot to monitor and control a
         PyTorch training process.
    # Arguments
        token: String, a telegram bot token
        user_id: Integer. Specifying a telegram user id will filter all incoming
                 commands to allow access only to a specific user. Optional,
                 though highly recommended.
    """

    def __init__(self, token, user_id=None):
        assert isinstance(token, str), 'Token must be of type string'
        assert user_id is None or isinstance(user_id, int), 'user_id must be of type int (or None)'

        self.token = token  # bot token
        self.user_id = user_id  # id of the user with access
        self.filters = None
        self.chat_id = None  # chat id, will be fetched during /start command
        self.bot_active = False  # currently not in use
        self.name = "TrainTrack"
        self._status_message = "No status report has been sent yet!"  # placeholder status message
        self.lr = None
        self.modify_lr = 1.0  # Initial lr multiplier
        self.verbose = True  # Automatic per epoch updates
        self.n_epoch = 0  # Number of epochs
        self.update_period = 1  # Send the message of every nth epoch
        self.prereport = True  # Activate pre-report as default
        self.stop_train_flag = False  # Stop training flag
        self.updater = None
        # Initialize status list
        self.status_list = []
        # Initialize loss and accuracy monitoring
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        # Enable logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Message to display on /start and /help commands
        self.startup_message = "Hello, this is TrainTrack! I will keep you updated on your training process.\n" \
                               " Send /help to see all of the options\n" \
                               " Send /start to activate automatic updates every epoch\n" \
                               " Send /set_period <n> to send the updates every n\'th epoch\n" \
                               " Send /toggle_prereport to turn on/off pre-report updates\n" \
                               " Send /quiet to stop getting automatic updates\n" \
                               " Send /getlr to query the current learning rate\n" \
                               " Send /setlr to manually change learning rate\n" \
                               " Send /dislr to disable the control of learning rate\n" \
                               " Send /status to get the latest results\n" \
                               " Send /plot to get a accuracy and loss convergence plot\n" \
                               " Send /stop_training to stop training process\n\n"

    def activate_bot(self):
        """ Function to initiate the Telegram bot """
        self.updater = Updater(self.token, use_context=True)  # setup updater
        dp = self.updater.dispatcher  # Get the dispatcher to register handlers
        dp.add_error_handler(self.error)  # log all errors
        print("TrainTrack has been initiated.\n"
              "Don\'t forget to start it on Telegram!\n")

        self.filters = Filters.user(user_id=self.user_id) if self.user_id else None
        # Command and conversation handles
        dp.add_handler(CommandHandler("start", self.start, filters=self.filters))  # /start
        dp.add_handler(CommandHandler("help", self.help, filters=self.filters))  # /help
        dp.add_handler(CommandHandler("getlr", self.get_lr, filters=self.filters))  # /get learning rate
        dp.add_handler(CommandHandler("dislr", self.disable_lr, filters=self.filters))  # /disable learning rate control
        dp.add_handler(CommandHandler("quiet", self.quiet, filters=self.filters))  # /stop automatic updates
        dp.add_handler(CommandHandler("status", self.status, filters=self.filters))  # /get status
        dp.add_handler(CommandHandler("plot", self.plot_loss, filters=self.filters))  # /plot loss
        dp.add_handler(CommandHandler("set_period", self.set_period, pass_args=True,
                                      filters=self.filters))  # /set frequency
        dp.add_handler(self.prereport_handler())  # toggle on/off pre-report updates
        dp.add_handler(self.lr_handler())  # set learning rate
        dp.add_handler(self.stop_handler())  # stop training
        dp.add_handler(MessageHandler(Filters.command, self.unknown))  # unknown command handler

        # Start the Bot
        self.updater.start_polling()
        self.bot_active = True

        # Uncomment next line while debugging
        # updater.idle()

    def stop_bot(self):
        """ Function to stop the bot """
        self.updater.stop()
        self.bot_active = False

    def start(self, update: Update, context: CallbackContext):
        """ Telegram bot callback for the /start command.
        Fetches chat_id, activates automatic epoch updates and sends startup message"""
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id
        self.verbose = True

    def help(self, update, context):
        """ Telegram bot callback for the /help command. Replies the startup message"""
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id

    def quiet(self, update, context):
        """ Telegram bot callback for the /quiet command. Stops automatic epoch updates"""
        self.verbose = False
        update.message.reply_text("Automatic epoch updates turned off.\n"
                                  "Send /start to turn epoch updates back on.")

    def error(self, update, context):
        """Log Errors caused by Updates."""
        self.logger.warning('Update "%s" caused error "%s"', update, context.error)

    def unknown(self, update, context):
        update.message.reply_text("Sorry! I didn't understand that command.")

    def send_message(self, txt):
        """ Function to send a Telegram message to user
         # Arguments
            txt: String, the message to be sent
        """
        assert isinstance(txt, str), 'Message text must be of type string'
        if self.chat_id is not None:
            self.updater.bot.send_message(chat_id=self.chat_id, text=txt)
        else:
            print('Send message failed, user did not send /start')

    def add_status(self, txt):
        """ Function to set a status message to be returned by the /status command """
        assert isinstance(txt, str), 'Status Message must be of type string'
        self._status_message = txt
        self.status_list.append(self._status_message)

    # Method for clearing status list
    def clr_status(self):
        self.status_list.clear()

    def update_epoch(self, n):
        assert isinstance(n, int), 'Number of epochs must be of type integer'
        self.n_epoch = n

    def update_message(self, msg):
        assert isinstance(msg, str), 'Status Message must be of type string'
        if self.verbose:
            if self.n_epoch % self.update_period == 0:
                self.send_message(msg)

    def update_prereport(self, msg):
        assert isinstance(msg, str), 'Status Message must be of type string'
        if self.prereport:
            self.update_message(msg)

    def set_period(self, update, context):
        self.chat_id = update.message.chat_id
        try:
            n_to_set = int(context.args[0])
            if n_to_set < 1:
                update.message.reply_text("Sorry, epochs don't go back in time!\n"
                                          "A positive integer was expected here.")
                return

            self.update_period = n_to_set
            update.message.reply_text("OK, updates will be sent every {} epoch(s).".format(self.update_period))
        except (IndexError, ValueError):
            update.message.reply_text("Usage: /set_period <positive integer>")

    def status(self, update, context):
        """ Telegram bot callback for the /status command. Replies with the latest status"""
        for state in self.status_list:
            self._status_message = self._status_message + "\n" + state
        update.message.reply_text(self._status_message)

        # Toggling pre-report updates

    def toggle_prereport(self, update, context):
        """ Telegram bot callback for the /toggle_prereport command. Displays verification message with buttons"""
        reply_keyboard = [['On', 'Off']]
        update.message.reply_text(
            "Toggle pre-report updates\n",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard))
        return 1

    def toggle_prereport_verify(self, update, context):
        """ Telegram bot callback for the /toggle_prereport command. Handle user selection as part of conversation"""
        is_sure = update.message.text  # Get response
        if is_sure == 'On':
            self.prereport = True
            update.message.reply_text('OK, turning on pre-report updates.', reply_markup=ReplyKeyboardRemove())
        elif is_sure == 'Off':
            self.prereport = False  # to allow changing your mind before stop took place
            update.message.reply_text('OK, turning off pre-report updates.', reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def cancel_prereport(self, update, context):
        """ Telegram bot callback for the /toggle_prereport command. Handle user cancellation as part of conversation"""
        if self.prereport:
            update.message.reply_text('User cancelled, pre-report updates are still on.',
                                      reply_markup=ReplyKeyboardRemove())
        elif not self.prereport:
            update.message.reply_text('User cancelled, pre-report updates are still on.',
                                      reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    def prereport_handler(self):
        """ Function to setup the callbacks for the /toggle_prereport command. Returns a conversation handler """
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('toggle_prereport', self.toggle_prereport, filters=self.filters)],
            states={1: [MessageHandler(Filters.regex('^(On|Off)$'), self.toggle_prereport_verify)]},
            fallbacks=[CommandHandler('cancel', self.cancel_prereport, filters=self.filters)])
        return conv_handler

    # Setting Learning Rate Callbacks:
    def get_lr(self, update, context):
        """ Telegram bot callback for the /getlr command. Replies with current learning rate"""
        if self.lr:
            update.message.reply_text("Current learning rate: " + str(self.lr))
        else:
            update.message.reply_text("Learning rate is not controlled by TrainTrack.\n"
                                      "To pass a learning rate, use the /setlr command.")

    def set_lr_front(self, update, context):
        """ Telegram bot callback for the /setlr command. Displays option buttons for learning rate multipliers"""
        if self.lr is None:
            self.lr = 1
            update.message.reply_text(
                'Learning rate is now under TrainTrack\'s control and is set to {}.'.format(self.lr))
        reply_keyboard = [['X0.1', 'X0.5', 'X0.67', 'X1.5', 'X2', 'X10']]  # possible multipliers
        # Show message with option buttons
        update.message.reply_text(
            'Change learning rate, multiply by a factor of: '
            '(Send /cancel to leave LR unchanged).\n\n',
            reply_markup=ReplyKeyboardMarkup(reply_keyboard))
        return 1

    def set_lr_back(self, update, context):
        """ Telegram bot callback for the /setlr command. Handle user selection as part of conversation"""
        options = {'X0.1': 0.1, 'X0.5': 0.5, 'X0.67': 0.67, 'X1.5': 1.5, 'X2': 2.0, 'X10': 10.0}  # possible multipliers
        self.modify_lr = options[update.message.text]  # User selection
        if self.lr is not None:
            self.lr *= self.modify_lr
            update.message.reply_text(" Learning rate will be multiplied by {0} on the beginning of next epoch!\n"
                                      "(New LR = {1:.4e})".format(str(self.modify_lr), self.lr * self.modify_lr),
                                      reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def cancel_lr(self, update, context):
        """ Telegram bot callback for the /setlr command. Handle user cancellation as part of conversation"""
        self.modify_lr = 1.0
        update.message.reply_text('OK, learning rate will not be modified on next epoch.',
                                  reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def lr_handler(self):
        """ Function to setup the callbacks for the /setlr command. Returns a conversation handler """
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('setlr', self.set_lr_front, filters=self.filters)],
            states={1: [MessageHandler(Filters.regex('^(X0.5|X0.1|X0.67|X1.5|X2|X10)$'), self.set_lr_back)]},
            fallbacks=[CommandHandler('cancel', self.cancel_lr, filters=self.filters)])

        return conv_handler

    def disable_lr(self, update, context):
        """ Telegram bot callback for the /dislr command. Stops TrainTrack from controlling the learning rate"""
        if self.lr is not None:
            update.message.reply_text("TrainTrack no longer controls the learning rate!")
            self.lr = None
        else:
            update.message.reply_text("TrainTrack doesn't control the learning rate!")

    # Stop training process callbacks
    def stop_training(self, update, context):
        """ Telegram bot callback for the /stoptraining command. Displays verification message with buttons"""
        reply_keyboard = [['Yes', 'No']]
        update.message.reply_text(
            'Are you absolutely sure?\n'
            'This will stop your training process!\n\n',
            reply_markup=ReplyKeyboardMarkup(reply_keyboard))
        return 1

    def stop_training_verify(self, update, context):
        """ Telegram bot callback for the /stoptraining command. Handle user selection as part of conversation"""
        is_sure = update.message.text  # Get response
        if is_sure == 'Yes':
            self.stop_train_flag = True
            update.message.reply_text('OK, stopping training after this epoch!\n'
                                      'Note that you can still cancel your request by sending '
                                      '/stop_training and replying \"No\".', reply_markup=ReplyKeyboardRemove())
        elif is_sure == 'No':
            self.stop_train_flag = False  # to allow changing your mind before stop took place
            update.message.reply_text('OK, canceling stop request!', reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def cancel_stop(self, update, context):
        """ Telegram bot callback for the /stoptraining command. Handle user cancellation as part of conversation"""
        self.stop_train_flag = False
        update.message.reply_text('OK, training will not be stopped.',
                                  reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    def stop_handler(self):
        """ Function to setup the callbacks for the /stoptraining command. Returns a conversation handler """
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('stop_training', self.stop_training, filters=self.filters)],
            states={1: [MessageHandler(Filters.regex('^(Yes|No)$'), self.stop_training_verify)]},
            fallbacks=[CommandHandler('cancel', self.cancel_stop, filters=self.filters)])
        return conv_handler

    # Cumulative methods for train/test losses and accuracies
    def cumulate_train_loss(self, train_loss):
        self.train_loss.append(train_loss)

    def cumulate_test_loss(self, test_loss):
        self.test_loss.append(test_loss)

    def cumulate_train_acc(self, train_acc):
        self.train_acc.append(train_acc)

    def cumulate_test_acc(self, test_acc):
        self.test_acc.append(test_acc)

    # Plot accuracies and losses (cumulative)
    def plot_loss(self, update, context):
        # Telegram bot callback for the /plot command. Replies with a convergence plot image
        if plt is None:
            # matplotlib isn't installed
            update.message.reply_text("Sorry, can't plot!\n"
                                      "Did you install \"matplotlib\" for Python?")
            return

        # set the figure up so it plots both loss and accuracy
        fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
        axes[0].set_title("Losses")
        axes[1].set_title("Accuracies")
        for ax in range(len(axes)):
            axes[ax].set_xlabel("n of epoch")
        axes[0].set_ylabel("loss")
        axes[1].set_ylabel("accuracy")
        fig.suptitle("TrainTrack Loss and Accuracy Plots")

        # Plot the data if enough epochs has passed (min_epoch can be adjusted as desired)
        # This prevents retrieving an empty plot
        min_epoch = 2
        if self.n_epoch > min_epoch:
            axes[0].plot(self.train_loss, 'r--')
            axes[0].plot(self.test_loss, 'b-')
            axes[0].legend(['Train Loss', 'Test Loss'])
            axes[1].plot(self.train_acc, 'r--')
            axes[1].plot(self.test_acc, 'b-')
            axes[1].legend(['Train Acc', 'Test Acc'])

            # Get the image of the plot
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            update.message.reply_photo(buffer)  # Send the image to the user
        else:
            update.message.reply_text("Not enough epochs has iterated yet for a proper plot.\n"
                                      "Plotting should be available after epoch {}.".format(min_epoch))
