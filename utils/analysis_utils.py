import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn.functional as F

def evaluate_model(model, dataloader, visualize=False,patient_id=101, slice_id=1):
    device = torch.device("cuda")
    model.eval()
    model = model.to(device)
    i = 0
    scores = pd.DataFrame(columns=['patient_id', 'slice_id', 'dice_avg', 'dice_lv', 'dice_rv', 'dice_myo'])

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        y_pred = torch.argmax(outputs[2], axis=1)

        if visualize:
            # Visualize the input image, ground truth mask, and predicted mask
            input_image = inputs[0].cpu().numpy().transpose(1, 2, 0)
            # convert into a single channel to visualize
            ground_truth_mask = torch.argmax(targets[0], dim=0)
            predicted_mask = y_pred.cpu().numpy().transpose(1, 2, 0)

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(input_image, cmap='gray')

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(ground_truth_mask, cmap='gray')

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(predicted_mask, cmap='gray')

            plt.show()

        # compute dice
        # convert to 4 channels to compare with gt, since gt has 4 channels
        y_pred_onehot = F.one_hot(y_pred, 4).permute(0, 3, 1, 2)

        dice = compute_dice(y_pred_onehot, targets)
        dice_lv = dice[3].item()
        dice_rv = dice[1].item()
        dice_myo = dice[2].item()
        # skip background for mean
        dice_avg = dice[1:].mean().item()

        scores.loc[i] = {
                'patient_id': patient_id,
                'slice_id': slice_id % 10 + 1,
                'dice_avg': dice_avg,
                'dice_lv': dice_lv,
                'dice_rv': dice_rv,
                'dice_myo': dice_myo
            }
        if slice_id == 20:
          patient_id += 1
          slice_id = 0
        slice_id += 1
        i+= 1

    return scores

def compute_dice(pred_y, y):
    """
    Computes the Dice coefficient for each class in the ACDC dataset.
    Assumes binary masks with shape (num_masks, num_classes, height, width).
    :param pred_y: predicted masks
    :param y: ground truth masks
    :return: dice scores for each class
    """
    epsilon = 1e-6
    num_classes = pred_y.shape[1]
    device = torch.device("cuda")
    dice_scores = torch.zeros((num_classes,), device=device)

    for c in range(num_classes):
        intersection = torch.sum(pred_y[:, c] * y[:, c])
        sum_masks = torch.sum(pred_y[:, c]) + torch.sum(y[:, c])
        dice_scores[c] = (2. * intersection + epsilon) / (sum_masks + epsilon)

    return dice_scores


def plot_bmi_dice(bmi, dice_type='dice_avg'):
    """
    Plot boxplot of dice scores for each BMI category
    :param bmi: dataframe containing dice scores and BMI category
    :param dice_type: dice score type to plot
    :return: None
    """

    y_labels = {'dice_avg': 'Average Dice Score',
               'dice_lv': 'Left Ventricle Dice Score',
               'dice_myo': 'Myocardium Dice Score',
               'dice_rv': 'Right Ventricle Dice Score'}
    y_label = y_labels[dice_type]

    sns.set(style="whitegrid")
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(x="bmi_category", y=dice_type, data=bmi, palette="Set3")
    ax.set_xlabel('BMI Category', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plt.show()