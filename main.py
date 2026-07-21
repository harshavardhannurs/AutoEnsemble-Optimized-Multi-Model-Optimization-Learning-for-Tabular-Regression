from models.ffn import pipelined_model
import utils.model_utils as mu

train_loss_across_folds, val_loss_across_folds, r2_score_across_folds = pipelined_model(
        dataset_name='bike_sharing_demand',
        batch_size=64,
        no_of_layers=2,
        no_of_units=128,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_norm=False,
        num_epochs=50,
        n_splits=5
    )

mu.plot_losses(train_loss_across_folds, val_loss_across_folds)

print("The current hyperparameter configuration yields an average R2 Score of ", mu.average_test_loss(r2_score_across_folds))