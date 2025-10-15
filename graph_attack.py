import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
folder = "femnist/results/attack/"
attack_levels = [10, 25, 50, 75, 100]
atk_client = [ "2", "5", "7", "10" ]

df = pd.read_csv("femnist/results/attack/fl_metrics_10.csv")

rounds = df['round'].unique()

# Filter attack clients
rounds_df = df[ (df['client_id'] != 'avg') & (df['client_id'].isin(atk_client)) ]

# Use one color per attack client
colors = plt.get_cmap('tab10', len(atk_client))

plt.figure( figsize = (9, 8) )
# 1. History of loss per client over rounds;
for idx, c in enumerate(atk_client):
    # plotting each rounds
    temp_df = rounds_df[ rounds_df['client_id'] == c ].copy()
    temp_df = temp_df.sort_values(by='round')

    # print(temp_df['client_id'])
    plt.plot(temp_df['round'], 
             temp_df['eval_loss'], 
             marker = 'o', 
             label = f'Client {c}', 
             color = colors(idx) )

plt.title('Client Evaluation Loss Per Round')
plt.xlabel('Round')
plt.ylabel('Evaluation Loss')
plt.legend(title = 'Client ID', bbox_to_anchor = (1.25, 1), loc = 'upper right')
plt.tight_layout()
plt.grid(True)
plt.savefig('femnist/results/attack/Client Evaluation Loss Per Round.png')
plt.show()

plt.figure( figsize = (8, 8) )
# 2. History of average loss among clients over rounds;
for idx, level in enumerate(attack_levels):
    file_path = f"{folder}/fl_metrics_{level}.csv"
    try:
        df = pd.read_csv(file_path)
        avg_df = df[ df['client_id'] == 'avg' ]
        plt.plot(avg_df['round'], 
                 avg_df['avg_loss'], 
                 marker = 'o', 
                 label = level,
                 color = colors(idx) )

    except Exception as e:
        print(f"Skipping {file_path}: {e}")

plt.title('Client Average Loss Per Round')
plt.xlabel('Round')
plt.ylabel('Average Loss')
plt.legend(title = 'Attack Level', bbox_to_anchor = (1.25, 1), loc = 'upper right')
plt.tight_layout()
plt.grid(True)
plt.savefig('femnist/results/attack/Client Average Loss Per Round.png')
plt.show()

plt.figure( figsize = (9, 8) )
# 3. History of evaluation accuracy per client over rounds;
for idx, c in enumerate(atk_client):
    # plotting each rounds
    temp_df = rounds_df[ rounds_df['client_id'] == c ].copy()
    temp_df = temp_df.sort_values(by='round')

    # print(temp_df['client_id'])
    plt.plot(temp_df['round'], 
             temp_df['eval_acc'], 
             marker = 'o', 
             label = f'Client {c}', 
             color = colors(idx) )

plt.title('Client Evaluation Accuracy Per Round')
plt.xlabel('Round')
plt.ylabel('Evaluation Accuracy')
plt.legend(title = 'Client ID', bbox_to_anchor = (1.25, 1), loc = 'upper right')
plt.tight_layout()
plt.grid(True)
plt.savefig('femnist/results/attack/Client Evaluation Accuracy Per Round.png')
plt.show()

plt.figure( figsize = (8, 8) )
# 4. History of average evaluation accuracy among clients over rounds.
for idx, level in enumerate(attack_levels):
    file_path = f"{folder}/fl_metrics_{level}.csv"
    try:
        df = pd.read_csv(file_path)
        avg_df = df[ df['client_id'] == 'avg' ]
        plt.plot(avg_df['round'], 
                 avg_df['avg_acc'], 
                 marker = 'o', 
                 label = level,
                 color = colors(idx) )

    except Exception as e:
        print(f"Skipping {file_path}: {e}")

plt.title('Client Average Accuracy Per Round')
plt.xlabel('Round')
plt.ylabel('Average Accuracy')
plt.legend(title = 'Attack Level', bbox_to_anchor = (1.25, 1), loc = 'upper right')
plt.tight_layout()
plt.grid(True)
plt.savefig('femnist/results/attack/Client Average Accuracy Per Round.png')
plt.show()

# 5. Final loss and accuracy dependence on the attack severity (flipping rate and the number of poisoned clients)
final_losses = []
final_accuracies = []

for level in attack_levels:
    file_path = f"{folder}/fl_metrics_{level}.csv"
    try:
        df = pd.read_csv(file_path)
        avg_df = df[df['client_id'] == 'avg']
        # Get the last round's metrics
        final_row = avg_df.iloc[-1]
        final_losses.append(final_row['avg_loss'])
        final_accuracies.append(final_row['avg_acc'])
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

# Plot final loss vs attack severity
plt.figure(figsize=(8, 6))
plt.plot(attack_levels, final_losses, marker='o', color='tab:red')
plt.title('Final Average Loss vs. Attack Severity')
plt.xlabel('Attack Severity (Flipping Rate or # Poisoned Clients)')
plt.ylabel('Final Average Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Final_Loss_vs_Attack_Severity.png")
plt.show()

# Plot final accuracy vs attack severity
plt.figure(figsize=(8, 6))
plt.plot(attack_levels, final_accuracies, marker='o', color='tab:blue')
plt.title('Final Average Accuracy vs. Attack Severity')
plt.xlabel('Attack Severity (Flipping Rate or # Poisoned Clients)')
plt.ylabel('Final Average Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Final_Accuracy_vs_Attack_Severity.png")
plt.show()