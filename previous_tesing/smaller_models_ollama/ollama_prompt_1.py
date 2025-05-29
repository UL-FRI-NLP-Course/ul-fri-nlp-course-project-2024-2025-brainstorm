from ollama import chat
from ollama import ChatResponse
import time

"""
- Assume the input data is a list of traffic eventsEach event ideally includes type (accident, jam, obstacle, roadwork), location, direction, description, and a timestamp.
- Prioritize recency: If multiple reports exist for the same location, use the information from the report with the latest timestamp relevant to the report time ([date] [time]).
- Extract key information: Identify the type of incident, precise location (road name/number, section, kilometer marker if available), direction of travel, and consequences (lane closures, delay times, type of obstruction).
- Handle missing information: If specific details (like exact delay times) are missing in the input, report the situation generally (e.g., "Zastoj je daljši," "Promet je močno oviran") rather than inventing details. Do not report incidents where essential information like location or type is completely missing.
"""

"""
EXAMPLE OF REPORT:
Here are examples demonstrating the desired structure and style:

"Prometne informacije        13.07 2023          14.30               2. program

Podatki o prometu.

Na gorenjski avtocesti proti Ljubljani je zaradi nesreče promet oviran pred priključkom Kranj-zahod.
Gost promet z zastoji je na več odsekih primorske avtoceste od Postojne proti Ljubljani. Opozarjamo na nevarnost naleta.
Zaradi del so zastoji tudi na mariborski vzhodni obvoznici med razcepom Dragučova in priključkom Maribor-center v obe smeri ter na severni ljubljanski obvoznici med razcepom Zadobrova in Tomačevim, prav tako v obe smeri. 
Na glavni cesti Ljubljana-Kočevje je zaradi nesreče promet v Velikih Laščah urejen izmenično enosmerno.
Na širšem območju Maribora je promet oviran zaradi močnega neurja. Voznike opozarjamo, da je ustavljanje pod nadvozi in v predorih nevarno in prepovedano."
"""

prompt_example = """You are an employee at RTVSlo responsible for generating radio traffic information. Your task is to create concise, accurate, and stylistically correct traffic reports in Slovenian based on provided input data.

INPUT DATA HANDLING:
- Use only active voice constructions (e.g., "Zaprta je..." [It is closed], not "Pas je zaprt" [The lane is closed]).
- Always mention locations with the direction of travel (e.g., "proti Ljubljani" [towards Ljubljana], "proti Kopru" [towards Koper]).
- Use the precise time format: "DD. M. YYYY ob HH:MM" (e.g., "29. 6. 2024 ob 14:00").
- Avoid English expressions; use Slovenian equivalents (e.g., "zastoj" instead of "delay", "gneča" instead of "traffic jam" unless specifically referring to a long standstill as "zastoj").
- Use standard phrases: "oviran promet" (obstructed traffic), "nastaja zastoj" (traffic jam is forming), "promet poteka počasi" (traffic is slow).
- Include specific traffic status descriptions when available: "promet poteka po odstavnem pasu" (traffic flows on the emergency lane), "zaprta sta vozni in počasni pas" (driving and slow lanes are closed).
- Use "prometne informacije" (traffic information) for the report title, not "prometno poročilo" (traffic report).
- Include specific delay times when available and significant: "Čas potovanja se podaljša za približno 30 minut". Use standard time references: "predvidoma do 20. ure" (expected until 8 PM).

CONTENT PRIORITY:
- Only include significant traffic incidents. Define "significant" as:
    - Accidents causing lane closures or delays estimated at 15 minutes or more, if not provided in data do not report.
    - Traffic jams resulting in delays estimated at 20 minutes or more, or those described as "daljši zastoj" (long jam) or "močan zastoj" (heavy jam) in the source data.
    - Obstacles completely blocking a lane or posing a direct safety hazard (e.g., large debris, stopped vehicle in a dangerous spot like a tunnel or before a bend).
    - Roadworks only if they are causing active, significant delays (approx. 15+ min) at the time of the report OR if they involve complete closures starting or ending today. Do not include routine, long-term roadworks unless they meet these criteria for the current reporting time.
    - If time estimates are not provided, use the following guidelines:
        - For accidents: Report if the situation is serious enough to cause significant delays or lane closures if provided in the data, if not do not report.
        - For traffic jams: Report if the situation is serious enough to cause significant delays or lane closures if provided in the data, if not do not report.
        - For obstacles: Report if the situation is serious enough to cause significant delays or lane closures if provided in the data, if not do not report.
        - For roadworks: Report if the situation is serious enough to cause significant delays or lane closures if provided in the dat), if not do not report.
- Filter based on report time: Exclude any incidents confirmed as resolved before the specified report time "[date] [time]". Ignore roadworks planned for the future or completed in the past relative to the report time.

FORMATTING:
- Use short, informative sentences focused on location and impact. Maintain a neutral, concise, factual, and clear tone appropriate.
- Start the entire report directly with "Prometne informacije [date] [time] za 1., 2. in 3. program Radia Slovenija." (Fill in the actual date and time). No preceding text.
- Immediately following the header line, include the standalone line "Podatki o prometu.".
- Ensure a single blank line follows "Podatki o prometu." before the first traffic item.
- Group all reports for a specific road together (e.g., all traffic jams on the A1 Primorska highway appear consecutively). 
- Keep paragraphs concise, focusing on one specific event or a set of closely related events on the same road section (max 2-3 sentences typically).
- Write report without including headings (like "Nesreče:", "Zastoji:", "Ovire:", "Vreme:", "Delo na cesti:", "Opozorila:"). Use the report structure as described below.
- End paragraphs describing accidents or dangerous obstacles with "Opozarjamo na nevarnost naleta." where appropriate (i.e., where traffic is stopped or slowed abruptly). Do not add this warning if the incident is just a slow-down without stationary vehicles or if the hazard is minor.
- Include only the most relevant and significant information. The goal is to provide a clear and concise summary of the traffic situation without overwhelming details.
- The report should be based on input data. Do not include any additional information or commentary outside the specified structure. 
- VERY IMPORTANT! - Avoid excessive repetition of events. If a road is closed due to an accident, do not repeat the same information in the Traffic jams or roadworks sections. Instead, summarize the situation in a single sentence. And do not report the same event multiple times in different sections (e.g., once as an accident and again as closed lane or traffic jam).


REPORT STRUCTURE (strict structure):
1. Header: "Prometne informacije      [date] [time]        1., 2. in 3. program"
2. Standalone Line: "Podatki o prometu."
3. Blank Line
4. Accidents (Significant accidents on the road.)
5. Traffic Jams (Only significant delays as defined above. Include obstacles causing delays.)
6. Roadworks (Only those causing significant delays today or full closures starting/ending today.)
7. Warnings / Restrictions / Reopenings (General warnings like weather impacts if severe, truck restrictions (e.g., weight/timing), information on major roads being reopened, buying of Vinjeta.)(Keep very short.)


INPUT DATA TO BE USED IN THE OUPUT REPORT:
 Timestamp: 2024-05-25 11:30:00
 Nesreče na cesti: Cesta Bistrica ob Sotli - Podsreda je zaprta pri Zagaju. Obvoz je možen za osebna vozila po lokalnih cestah. 
 Zastoji: Na ljubljanski obvoznici proti Primorski:- Podutik - Kozarje, zamuda 20 minut;- med Vičem in Kozarjami zamuda 5 minut.Na primorski avtocesti med Ljubljano in Vrhniko proti Kopru, zamuda 15 minut. Na štajerski avtocesti med Framom in Slovensko Bistrico proti Ljubljani, zamuda približno 5 - 10 inut. Na hitri cesti Koper - Škofije pred mejnim prehodom proti Italiji, nesreča na Italijanski strani. Na cestah Ljubljana - Brezovica, Šmarje - Koper in Lucija - Strunjan. 
 Ovire: Na vipavski hitri cesti je med Vipavo in Podnanosom proti Razdrtemu zaprt vozni pas, okvara vozila. Na podravski avtocesti je oviran promet pri priključku Letališče Maribor proti Mariboru, povožena žival.   
 Delo na cesti: Na gorenjski avtocesti med Kosezami in Šentvidom promet poteka po dveh zoženih pasovih v obe smeri. Hkrati so zaprti:- uvoz Šentvid, s Celovške ceste v predor proti Kosezam;- uvoz Podutik, s Podutiške ceste proti Kranju;- na razcepu Koseze krak iz smeri severne ljubljanske obvoznice proti Gorenjski, obvoz po Celovški cesti ali preko priključka Brdo.Cesta Lesce - Kamna Gorica - Lipnica bo v Kamni Gorici zaprta do nedelje, 26. maja, do 19. ure.Popolne zapore na cesti Litija - Zagorje, pri Šklendrovcu, ta vikend ne bo.Več o delovnih zaporah v prometni napovedi.
 Opozorila: 
 Vreme: 
 
 EXECUTION:
Using the provided data, compose a report in Slovenian language that could be read on the radio, strictly following all the above instructions, structure, and constraints. Stick to the provided examples as closly as possible in terms of formatting. If the report is over 800 characters you will be fired as a RTVSlo reporter. Wait for the data input.
 """

response: ChatResponse = chat(model='deepseek-llm:7b', messages=[
  {
    'role': 'user',
    'content': prompt_example,
  },
])
print(response['message']['content'])
print(response.message.content)

prometno_porocilo = response['message']['content']

timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
file_name = "prometno_porocilo_" + timestamp + ".txt"

with open(file_name, "w", encoding="utf-8") as f:
    f.write(prometno_porocilo)
