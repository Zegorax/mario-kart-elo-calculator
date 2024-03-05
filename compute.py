import pandas as pd
import numpy as np

def prob(diff):
    return 1/(1 + (10**(-diff/400)))

def compute_elo(df, players) :

    score = df.copy()
    score['Rank'] = score['Pts'].rank(method='dense', ascending=False).astype(int)

    score = pd.merge(score[['Challenger', 'Rank', 'Date']], players['Elo'], on = 'Challenger')
    score.set_index('Challenger', inplace=True)

    # Creating an empty DataFrame for the 2D matrix of probabilities
    probs = pd.DataFrame(index=score.index, columns=score.index)

    # Creating an empty DataFrame for the 2D matrix of results (win/loss/draw)
    wins = pd.DataFrame(index=score.index, columns=score.index)

    game_result = pd.DataFrame(columns=['Winner', 'Loser', 'Count'])

    # Populating the matrices with the 
    for i in score.index:
        for j in score.index:
            # the probability of win is computed using the prob function of the Elo score difference between the players
            probs.at[i, j] = prob(score.at[i, 'Elo'] - score.at[j, 'Elo'])
            # If the player won, he gets a 1, if draw then 0.5 if loss then 0
            wins.at[i, j] = (1 - np.sign(score.at[i, 'Rank'] - score.at[j, 'Rank']))/2
            # Compute the winned matches 
            if i != j :
                game_result.loc[len(game_result)] = [i, j, wins.loc[i,j]]
    
    K = 32

    new_elo = pd.DataFrame((score['Elo'] + K * np.sum(wins - probs, axis=1)).astype('int64'), columns=['Elo'])

    # Updating the player dataframe
    # 1. Update 'Elo' score
    players.loc[new_elo.index, 'Elo'] = new_elo['Elo']

    # 2. Increment 'Nb_matches'
    players.loc[new_elo.index, 'Nb_matches'] += 1

    # 3. Update 'Last_update'
    players.loc[new_elo.index, 'Last_update'] = score['Date']

    df_elo = pd.merge(df[['Date', 'Challenger', 'Round', 'Pts']], new_elo, on='Challenger')
    df_elo.index = df.index

    return players, df_elo, game_result

def main():
    # Load the tournaments
    tournaments = pd.read_excel('Data/MK8_Tournaments.xlsx', sheet_name='Scores')
    tournaments = tournaments.copy()
    tournaments['Date'] = tournaments['Date'].dt.strftime('%Y%m%d').astype(int)
    tournaments['Elo'] = 0


    # Create the players dataframe with initial Elo score of 1000
    players = pd.DataFrame(tournaments['Challenger'].unique(), columns=['Challenger'])

    players['Elo'] = 1000
    players['Nb_matches'] = 0
    players['Last_update'] = tournaments['Date'].min()

    players = players.set_index('Challenger')

    dates = tournaments['Date'].sort_values().unique().tolist()

    results: any = None

    for tournament_date in dates:
        tournament = tournaments[tournaments['Date'] == tournament_date]
        
        games = tournament['Round'].sort_values().unique().tolist()

        for game in games:
            players, df_elo, game_result = compute_elo(df=tournament[tournament['Round'] == game], players=players)
            
            tournaments.loc[df_elo.index, 'Elo'] = df_elo['Elo']
            game_result['Date'] = tournament_date
            game_result['Round'] = game

            results = pd.concat([results, game_result], ignore_index=True)
        
        print('Elo score computed for tournament: ' + str(tournament_date))

    players.sort_values('Elo', ascending=False)

    print(results)

    results.to_excel('results/results.xlsx', sheet_name='Scores', index=False)

if __name__ == "__main__":
    main()