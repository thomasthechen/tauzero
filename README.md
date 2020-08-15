# Python Chess AI
## Two models implemented:

(1) Minimax Agent w/ board-value neural network (basic)

(2) Monte Carlo Tree Search w/ move network (in style of AlphaZero)

## How to play:

To run web app locally (where you can play chess against the agent on an actual chess GUI), you need to export the environment variables in setup.txt, and need to create a database in postgres called testdb; then you can run python play_chess.py

To just view the monte carlo agent play against itself, you can just run python selfplay.py

You can also play against the Minimax agent with an actual GUI at <span style="background-color: #FFFF00">playtauzero.herokuapp.com</span>. Note the version deployed on heroku is simpler with less search power, though.

## Other components of the project:

Colab notebook for training models: https://colab.research.google.com/drive/1pbMSAsHiy0PuZ7Oak-lhZenrRcS6oIgV#scrollTo=jpLujOJVF8Lz

Trained Models: https://drive.google.com/drive/folders/1n0ioGu-UHqMpdZIfmvMLmX-WL3_kTinz

## A few resources used in this project:

### Articles and Papers:

Basic Minimax: https://www.freecodecamp.org/news/simple-chess-ai-step-by-step-1d55a9266977/

Basics of Monte Carlo: http://matthewdeakos.me/2018/03/10/monte-carlo-tree-search/?fbclid=IwAR3qmj9--m3s_0iVHateEMdjJBSbG9o4tHmontUL1hoXgt0sSodyRbFoa1U

AlphaGo Zero Paper: https://www.researchgate.net/publication/320473480_Mastering_the_game_of_Go_without_human_knowledge


### Coding resources:

Python chess library: https://python-chess.readthedocs.io/en/latest/index.html

Chess GUI credits: https://github.com/geohot/twitchchess

Heroku + Flask + Postgres tutorial: https://medium.com/@dushan14/create-a-web-application-with-python-flask-postgresql-and-deploy-on-heroku-243d548335cc

## Project Contributers

Thomas Chen `thomasthechen`
Collin Wang `collinwa`
Andrew Chen `archen2019`

