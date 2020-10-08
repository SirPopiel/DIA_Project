Data Intelligence Applications project
Course held @ Politecnico di Milano by Nicola Gatti and Alessandro Nuara

Pricing and Advertising

Developed by:
Federico Capello,
Filippo Fedeli,
Gianmarco Genalti,
Jean Paul Guglielmo Baroni,
Carlo Augusto Vitellio


The text of the project is in the file Text.txt
Every point of the request has its corresponding folder.
In order to run everything you need a compiler for Python files and Jupyter notebooks are provided for the last two points

- P2 and P3 are developed in a unique folder. You can execute each point in main.py.
If you want to change some initial assumption (included the choice between P2 and P3) you can do that from the file data.py. Here you can also find the real conversion rate curves used in our experiments, together with any assumption we made.

- P4 and P5 are developed in a unique folder. There's an input boolean parameter called 'split_allowed' which determines the approach to undertake: pricing with aggregated subcampagins if False or pricing with splitted subcampaigns if True.

- P6 and P7 are developed in two separate folders. Relevant inputs are: `n_arms_ads`  (number of arms for advertising), `n_arms_pricing` (number of arms for pricing), `N` (number of experiments run), `T` (Runtime) and `sigma` (standard deviation)
