import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv("femnist/fl_metrics.csv")

# print(df.head()) # View the first few rows
# print(df.columns) # View column names

rounds = df['round'].unique()
rounds_df = df[ df['client_id'] != 'avg' ]
avg_df = df[ df['client_id'] == 'avg' ]

colors = plt.get_cmap('tab20', len(rounds))

# 1. History of loss per client over rounds;
for idx, r in enumerate(rounds):
    # plotting each rounds
    temp_df = rounds_df[ rounds_df['round'] == r ].copy()

    # sorting the client_id
    temp_df['client_id'] = temp_df['client_id'].astype(int)
    temp_df = temp_df.sort_values( by = 'client_id' )

    # print(temp_df['client_id'])
    plt.plot(temp_df['client_id'], 
             temp_df['eval_loss'], 
             marker = 'o', 
             label = f'Round {r}', 
             color = colors(idx) )

plt.title('Client Evaluation Loss Per Round')
plt.xlabel('Client ID')
plt.ylabel('Evaluation Loss')
plt.xticks(temp_df['client_id'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('femnist/results/no_attack/Client Evaluation Loss Per Round.png')
plt.show()

# 2. History of average loss among clients over rounds;
plt.plot(avg_df['round'], 
         avg_df['avg_loss'], 
         marker = 'o', 
         label = f'Round {r}', 
         color = colors(idx) )
plt.title('Client Average Loss Per Round')
plt.xlabel('Round')
plt.ylabel('Average Loss')
plt.xticks(avg_df['round'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('femnist/results/no_attack/Client Average Loss Per Round.png')
plt.show()


# 3. History of evaluation accuracy per client over rounds;
for idx, r in enumerate(rounds):
    # plotting each rounds
    temp_df = rounds_df[ rounds_df['round'] == r ].copy()

    # sorting the client_id
    temp_df['client_id'] = temp_df['client_id'].astype(int)
    temp_df = temp_df.sort_values( by = 'client_id' )

    # print(temp_df['client_id'])
    plt.plot(temp_df['client_id'], 
             temp_df['eval_acc'], 
             marker = 'o', 
             label = f'Round {r}', 
             color = colors(idx) )

plt.title('Client Evaluation Accuracy Per Round')
plt.xlabel('Client ID')
plt.ylabel('Evaluation Accuracy')
plt.xticks(temp_df['client_id'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('femnist/results/no_attack/Client Evaluation Accuracy Per Round.png')
plt.show()

# 4. History of average evaluation accuracy among clients over rounds.
plt.plot(avg_df['round'], 
         avg_df['avg_acc'], 
         marker = 'o', 
         label = f'Round {r}', 
         color = colors(idx) )
plt.title('Client Average Accuracy Per Round')
plt.xlabel('Round')
plt.ylabel('Average Accuracy')
plt.xticks(avg_df['round'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('femnist/results/no_attack/Client Average Accuracy Per Round.png')
plt.show()