import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
folder = "femnist/results/attack/"
attack_levels = [10, 25, 50, 75, 100]

pid_df = pd.read_csv("femnist/results/attack/fl_metrics_pid_100.csv")
exclusions_df = pd.read_csv("femnist/results/attack/exclusions_fl_metrics_pid_100.csv")

pid_clients_df = pid_df[ pid_df["client_id"] != "avg" ].copy()
pid_average_df = pid_df[ pid_df["client_id"] == "avg" ].copy()
exclusion_list_df = exclusions_df[ ( exclusions_df["permanently_excluded"] == True ) | ( exclusions_df["excluded"] == True ) ].copy()

# print (exclusion_list_df)

clients = pid_clients_df['client_id'].unique()

# Use one color per client
colors = plt.get_cmap( 'tab20', len( clients ) )

# 1. History of loss per client over rounds;
plt.figure( figsize = (10, 8) )

for idx, c in enumerate(clients):
    # Filter data for the client
    temp_df = pid_clients_df[ pid_clients_df["client_id"] == c ].copy()
    excluded_df = exclusion_list_df[ exclusion_list_df["client_id"] == c ].copy()

    # Sort by round before plotting
    temp_df = temp_df.sort_values( by = 'round' )

    if not excluded_df.empty:
        plt.plot(
            temp_df['round'], 
            temp_df['eval_loss'], 
            marker = 'o', 
            label = c + " Removed", 
            color = colors(idx) 
        )
        
        removal_round = int(excluded_df["round"].iloc[0]) - 1
        match = temp_df[ temp_df["round"].astype(int) == removal_round ]
        
        # print(temp_df["round"])
        # print(removal_round)
        if not match.empty:
            value = match["eval_loss"].iloc[0]

            # print(value)

            plt.scatter( 
                removal_round,
                value,
                color = 'tab:red',
                marker = 'x',
                s = 100,
                zorder = 5,
            )
    else:
        plt.plot(
            temp_df['round'], 
            temp_df['eval_loss'], 
            marker = 'o', 
            label = c, 
            color = colors(idx) 
        )
    
plt.title("Client Evaluation Loss Per Round (PID with Removal)")
plt.xlabel("Round")
plt.ylabel("Evaluation Loss")
plt.legend(title = "Client ID", bbox_to_anchor = (1.25, 1), loc = "upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Client_Eval_Loss_Per_Round_PID_Removal.png")
plt.show()

# 2. History of average loss among clients over rounds;
plt.figure( figsize = (9, 8) )

plt.plot(
    pid_average_df['round'], 
    pid_average_df['avg_loss'], 
    marker = 'o', 
    color = colors(idx) 
)

plt.title('Client Average Loss Per Round')
plt.xlabel('Round')
plt.ylabel('Average Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Client Average Loss Per Round (PID).png")
plt.show()


# 3. History of evaluation accuracy per client over rounds;
plt.figure( figsize = (10, 8) )

for idx, c in enumerate(clients):
    # Filter data for the client
    temp_df = pid_clients_df[ pid_clients_df["client_id"] == c ].copy()
    excluded_df = exclusion_list_df[ exclusion_list_df["client_id"] == c ].copy()

    # Sort by round before plotting
    temp_df = temp_df.sort_values( by = 'round' )

    if not excluded_df.empty:
        plt.plot(
            temp_df['round'], 
            temp_df['eval_acc'], 
            marker = 'o', 
            label = c + " Removed", 
            color = colors(idx) 
        )
        
        removal_round = int(excluded_df["round"].iloc[0]) - 1
        match = temp_df[ temp_df["round"].astype(int) == removal_round ]
        
        # print(temp_df["round"])
        # print(removal_round)
        if not match.empty:
            value = match["eval_acc"].iloc[0]

            # print(value)

            plt.scatter( 
                removal_round,
                value,
                color = 'tab:red',
                marker = 'x',
                s = 100,
                zorder = 5,
            )
    else:
        plt.plot(
            temp_df['round'], 
            temp_df['eval_acc'], 
            marker = 'o', 
            label = c, 
            color = colors(idx) 
        )
    
plt.title("Client Evaluation Accuracy Per Round (PID with Removal)")
plt.xlabel("Round")
plt.ylabel("Evaluation Accuracy")
plt.legend(title = "Client ID", bbox_to_anchor = (1.25, 1), loc = "upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Client_Eval_Acc_Per_Round_PID_Removal.png")
plt.show()

# 4. History of average evaluation accuracy among clients over rounds.
plt.figure( figsize = (9, 8) )

plt.plot(
    pid_average_df['round'], 
    pid_average_df['avg_acc'], 
    marker = 'o', 
    color = colors(idx) 
)

plt.title('Client Average Accuracy Per Round')
plt.xlabel('Round')
plt.ylabel('Average Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Client Average Accuracy Per Round (PID).png")
plt.show()

# 5. Loss and accuracy dependence on the attack severity
plt.figure( figsize = (12, 8) )

for idx, level in enumerate(attack_levels):
    file_path = f"{folder}/fl_metrics_pid_{level}.csv"

    try:
        df = pd.read_csv(file_path)
        avg_df = df[df["client_id"] == "avg"]  # only average metrics
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

    if not avg_df.empty:
        plt.plot(
            avg_df['round'], 
            avg_df['avg_loss'], 
            marker = 'o', 
            label = f"{level}% Evaluation Loss", 
            color = colors(idx) 
        )

plt.title('Loss Dependence on the Attack Severity')
plt.xlabel('Round')
plt.ylabel('Average Loss')
plt.legend(title = "Attack Severity", bbox_to_anchor = (1.25, 1), loc = "upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Loss Dependence (PID).png")
plt.show()

plt.figure( figsize = (12, 8) )

for idx, level in enumerate(attack_levels):
    file_path = f"{folder}/fl_metrics_pid_{level}.csv"

    try:
        df = pd.read_csv(file_path)
        avg_df = df[df["client_id"] == "avg"]  # only average metrics
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

    if not avg_df.empty:
        plt.plot(
            avg_df['round'], 
            avg_df['avg_acc'], 
            marker = 'o', 
            label = f"{level}% Evaluation Accuracy",
            color = colors(idx) 
        )

plt.title('Accuracy Dependence on the Attack Severity')
plt.xlabel('Round')
plt.ylabel('Average Accuracy')
plt.legend(title = "Attack Severity", bbox_to_anchor = (1.25, 1), loc = "upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{folder}/Accuracy Dependence (PID).png")
plt.show()