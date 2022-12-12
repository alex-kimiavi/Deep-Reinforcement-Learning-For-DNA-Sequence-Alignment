import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_string('game', 'Pairwise', """Alignment mode selection""")
tf.app.flags.DEFINE_string('model_name', "C51", """Name of the model (DQN or SSD)""")
tf.app.flags.DEFINE_string('network_set',"C51", """Network strategy setting (standard or SSD)""")
tf.app.flags.DEFINE_string('exploration', "e-greedy", """Name of the exploration strategy (e-greedy, boltzmann, bayesian""")
tf.app.flags.DEFINE_string('strategy',"test_pair", """Test strategy setting""")
tf.app.flags.DEFINE_boolean('resume', True, """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('show_training', False, """Show windows with workers training""")
tf.app.flags.DEFINE_boolean('show_align', False, """Show alignment figure""")
tf.app.flags.DEFINE_boolean('print_align', False, """Print alignment results""")
tf.app.flags.DEFINE_boolean('display_process', True, """Show progress state""")
tf.app.flags.DEFINE_boolean('use_GPU', True, """Usage of GPU or CPU""")
tf.app.flags.DEFINE_boolean('test_identity', True, """Tests the ratio of exact match results versus the optimal alignment""")
