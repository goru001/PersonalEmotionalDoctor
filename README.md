# PersonalEmotionalDoctor

My vision for this chat bot, named Personal Emotional Doctor, is of a friend who gives personalized counselling without being emotionally biased and judgemental, which are rare qualities in humans these days

This is a Research project and is a Work in Progress.

## Background and Vision

Depression is a state of low mood and aversion to activity that affects person's thoughts. According to WHO, depression will be the 2nd leading cause of disease by 2020. Moreover, depression is an extreme stage of what we experience everyday, i.e sadness. A good counselling when you're low, can do wonders; a motivational speech can get you started towards your fightback, a good relationship advice can help in relationship crisis. We do have people around us who motivate us in tough times with their experiences of life, but,they end up being  judgemental and emotionally biased, because of which we’re not able to completely share our feelings with humans, and are left fighting sadness alone.
A machine can't be emotionally biased and judgemental. We plan on building a bot, that gives personalized counselling to the "sad" person. The bot will maintain a repository of motivational, psychological content available on the internet, through religious scriptures, through psychological books and will interact with the sad person using that knowledge to heal him. The "sad" person can trust the machine, because it won't tell his problem to anyone else, nor would it get judgemental, as normal human beings do.

## Dataset 

* You can find the current dataset of conversations in `dataset` folder along with the pre-processing scripts. I've collected this dataset from
Sister Shivani's book Happiness Unlimited , which is a conversational adaptation from the internationally acclaimed TV Series "Awakening with BrahmaKumaris" series [Happiness Unlimited](https://www.youtube.com/playlist?list=PLCE9046E85D3918D8).
* The knowledge base `dataset/englishAvyaktMurlis.pkl` for the bot comes from "Avyakt Murlis", which I scraped using the script `dataset/get-avyakt-murlis.ipynb`. This is because Sister Shivani says all her knowledge comes from these Avyakt Murlis.


## Model

The current model, which you can find in `model.py`,  is an encoder-decoder model using AWD-LSTMs, with
"attention over questions" + "attention over knowledge base" + "feedback of previous output"

This is inline with the aim to build an end to end conversational system instead of relying on manual intervention to extract intents, entities etc like in traditional NLP to build a chat-bot.

## TODOs

* Currently the conversational dataset is very small. We have only 300
conversation pairs between Sister Shivani and Suresh Oberoi. Sister Shivani has lots of other series as well, which you can check out [here](https://www.youtube.com/results?search_query=bk+shivani+series).
So, the first task is to get those conversation pairs.

* The code doesn't run parallely yet, because of "feedback of previous output". Need
to make it parallel by using a trick similar to what we use in Language Models.