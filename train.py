import time
import json
import tqdm
import matcher.semantic_matcher as sm

for n in [7,5,3]:
    with open('scenario.json') as json_file:
        scenario = json.load(json_file)
        #print(f'{scenario}')

        semantiMatcher = sm.SemanticMathcer(path='corpus', n=n, latent=True)

        # Train for the service description
        services = scenario['services']
        for s in tqdm.tqdm(services):
            semantiMatcher.add(s)

        # Train for the queries
        semantiMatcher.buildIdx()

        # For each test
        tests = ['queries m2m', 'queries one-error', 'queries two-errors-one-word',
        'queries two-errors-two-words', 'queries one-synonym', 'queries two-synonyms',
        'queries three-synonyms', 'queries four-synonyms']

        # Query the system and learn the word profile
        for t in tqdm.tqdm(tests):
            queries = scenario[t]
            for q in tqdm.tqdm(queries, leave=False):
                services = semantiMatcher.match(q['query'])
        