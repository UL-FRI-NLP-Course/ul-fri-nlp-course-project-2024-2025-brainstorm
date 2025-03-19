# Natural language processing course: Automatic generation of Slovenian traffic news for RTV Slovenija

INSTRUCTIONS:

This project aims to use an existing LLM, »fine-tune« it, leverage prompt engineering techniques to generate short traffic reports. You are given Excel data from promet.si portal and your goal is to generate regular and important traffic news that are read by the radio presenters at RTV Slovenija. You also need to take into account guidelines and instructions to form the news. Currently, they hire students to manually check and type reports that are read every 30 minutes.

Methodology

6. Literature Review: Conduct a thorough review of existing research and select appropriate LLMs for the task. Review and prepare an exploratory report on the data provided.

7. Initial solution: Try to solve the task initially only by using prompt engineering techniques.

8. Evaulation definition: Define (semi-)automatic evaluation criteria and implement it. Take the following into account: identification of important news, correct roads namings, correct filtering, text lengths and words, ...

9. LLM (Parameter-efficient) fine-tuning: Improve an existing LLM to perform the task automatically. Provide an interface to do an interactive test.

10. Evaluation and Performance Analysis: Assess the effectiveness of each technique by measuring improvements in model performance, using appropriate automatic (P, R, F1) and human evaluation metrics.

Recommended Literature

· Lab session materials · RTV Slo data: LINK (zip format). The data consists of:

o Promet.si input resources (Podatki - PrometnoPorocilo_2022_2023_2024.xlsx).

o RTV Slo news texts to be read through the radio stations (Podatki - rtvslo.si).

o Additional instructions for the students that manually type news texts (PROMET, osnove.docx, PROMET.docx).


EXTRA INSTRUCTIONS:

Our group has a few questions regarding the project:

    Main Objective – What is the primary goal? Is the aim to generate traffic news articles from Excel documents? Are the RTF files in individual folders intended as reference examples (ground truth)?
    Formatting Rules – The two Word documents in the root folder seem to contain guidelines for formatting news articles. Are they meant to define the required format?
    News Linking – We noticed that some news articles reference previous ones. Should our solution detect and establish references to prior news articles, or is it sufficient to generate news solely based on individual Excel rows?
    LLM Selection – Which language model should we use? Do we have full flexibility in choosing one (or trying several) and running pre-trained models on ARNES? We assume the OpenAI API is not a suitable option for this project.
    RTV SLO News for Radio Broadcast (Data – rtvslo.si) – This was listed as an additional data source. Could you clarify what was meant by this? Was the intention to crawl rtvslo.si and extract traffic news data?

    Main objective: You have input data (Podatki - PrometnoPorocilo_2022_2023_2024.xlsx) and goal is to generate outputs (folder Podatki - rtvslo.si). Beside that you need to analyze data first to find out correlations, when traffic reports are urgent, ... You can think that this is an automation of a job, currently done manually. You also have documents with guidelines - these are instructions for the human writers. 
    Formatting Rules: Yes, take those also into account.
    News Linking: See my first answer. All the projects are open, so your job is to do quality work, which you can justify. In real-world scenarios, your instructions will not be often precisely defined. If you will just generate text for specific rows without context and understanding, the results will not be best. So, try to infer first, which rows are correlated to (a) highly-important traffic announcements that are prepared immediately, and (b) standard half-hour reports.
    LLM Selection: APIs can be directly used by anyone. Of course you can start with OpenAI or some pre-trained models (e.g., using Ollama library), and do some prompt engineering. The goal in this course is to do some NLP work, not only "use models or APIs" - you could do that easily also without taking attending this course. We will cover topics of fine-tuning the models in the following weeks during the lab sessions. 
    RTV SLO News for Radio Broadcast (Data – rtvslo.si): As said, projects are open. If you can find also some other sources that are relevant, you can improve your results, even better. For example: maybe weather, calendar data (holidays), ... can help you with the project - you need to be creative. We will discuss these during defenses also, so you will get an impression of your work during the semester. 
