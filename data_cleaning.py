import re
import csv

years = '^\d+-\d+'

# Clean the All NBA data from the nba.com site
# https://www.nba.com/history/awards/all-nba-team

with open('data/all_nba_teams.csv') as text:
    with open('data/all_nba_teams_clean.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        year = ''
        team = 0
        for line in text:
            name = ''
            print(line)
            if re.search(years, line):
                year = float(line[:4].replace('\n', '')) + 1

            if line.startswith('FIRST'):
                team = 1
            elif line.startswith('SECOND'):
                team = 2
            elif line.startswith('THIRD'):
                team = 3

            if line.find(':') != -1:
                if line.find(',') != -1:
                    name = line[line.index(':') + 2:line.index(',')]
                    print(name)
                else:
                    name = line[line.index(':') + 2:].replace('\n', '')
            print(year, team)
            if name != '':
                writer.writerow([year, team, name])
