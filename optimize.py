import time
import json
import tqdm
import optuna
import logging

import semantic.dp as dp
import matcher.semantic_matcher as sm
import matcher.metrics as mm


logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)


def train_dpw_model(path, limit, n, latent, k, c):
    jt = 0.0 #trial.suggest_float('jt', 0.0, 1.0, step=0.01)
    st = 0.15 #trial.suggest_float('st', 0.0, 1.0, step=0.01)

    cutoff = {
        'pareto20': dp.Cutoff.pareto20,
        'pareto80': dp.Cutoff.pareto80,
        'none': dp.Cutoff.none,
        'knee': dp.Cutoff.knee
    }

    semanticMatcher = sm.SemanticMathcer(path=path, limit=limit, model='dpw',
    jt=jt, st=st, n=n, latent=latent, k=k, c=cutoff[c])

    relevant = []
    received_cosine = []
    received_jaccard = []
    with open('scenario.json') as json_file:
        scenario = json.load(json_file)
        
        # Train for the service description
        services = scenario['services']
        for s in tqdm.tqdm(services, leave=False):
            semanticMatcher.add(s)

        # Train for the queries
        semanticMatcher.buildIdx()
    
    return semanticMatcher


def train_dpwc_model(path, limit, n, latent, k, c, kl):
    jt = 0.0 #trial.suggest_float('jt', 0.0, 1.0, step=0.01)
    st = 0.15 #trial.suggest_float('st', 0.0, 1.0, step=0.01)

    cutoff = {
        'pareto20': dp.Cutoff.pareto20,
        'pareto80': dp.Cutoff.pareto80,
        'none': dp.Cutoff.none,
        'knee': dp.Cutoff.knee
    }

    semanticMatcher = sm.SemanticMathcer(path=path, limit=limit, model='dpwc', 
    jt=jt, st=st, n=n, latent=latent, k=k, c=cutoff[c], kl=kl)

    relevant = []
    received_cosine = []
    received_jaccard = []
    with open('scenario.json') as json_file:
        scenario = json.load(json_file)
        
        # Train for the service description
        services = scenario['services']
        for s in tqdm.tqdm(services, leave=False):
            semanticMatcher.add(s)

        # Train for the queries
        semanticMatcher.buildIdx()
    
    return semanticMatcher


def semantic_dpw_objective(trial, path, limit):
    n = trial.suggest_categorical('n', [3,5,7])
    latent = trial.suggest_categorical('latent', [True, False])
    k = trial.suggest_categorical('k', [2,3,4,5])
    c = trial.suggest_categorical('c', ['pareto20', 'pareto80', 'knee'])
    
    jt = 0.0 #trial.suggest_float('jt', 0.0, 1.0, step=0.01)
    st = 0.15 #trial.suggest_float('st', 0.0, 1.0, step=0.01)

    cutoff = {
        'pareto20': dp.Cutoff.pareto20,
        'pareto80': dp.Cutoff.pareto80,
        'none': dp.Cutoff.none,
        'knee': dp.Cutoff.knee
    }

    semantiMatcher = sm.SemanticMathcer(path=path, limit=limit, model='dpw',
    jt=jt, st=st, n=n, latent=latent, k=k, c=cutoff[c])

    relevant = []
    received_cosine = []
    received_jaccard = []
    with open('scenario.json') as json_file:
        scenario = json.load(json_file)
        
        # Train for the service description
        services = scenario['services']
        for s in tqdm.tqdm(services, leave=False):
            semantiMatcher.add(s)

        # Train for the queries
        semantiMatcher.buildIdx()

        # For each test
        tests = ['queries m2m', 'queries one-error', 'queries two-errors-one-word',
        'queries two-errors-two-words', 'queries one-synonym', 'queries two-synonyms',
        'queries three-synonyms', 'queries four-synonyms']

        # Query the system and learn the word profile
        for t in tqdm.tqdm(tests, leave=False):
            queries = scenario[t]
            for q in tqdm.tqdm(queries, leave=False):
                services = semantiMatcher.match(q['query'])

                queryId = int(q['id']) % 100
                relevant.append([queryId])
                received_cosine.append([i for i, _ in services['cosine']])
                received_jaccard.append([i for i, _ in services['jaccard']])
    
    return mm.mean_average_precision(relevant, received_cosine)#,mm.mean_average_precision(relevant, received_jaccard))


def train_semantic_dpw_matcher(path:str='corpus', limit:int=300):
    # Running the Optuna optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: semantic_dpw_objective(trial, path, limit), n_trials=100, show_progress_bar=True)

    # Get the best model and hyperparameters
    best_params_optuna = study.best_params

    logger.info(best_params_optuna)
    return best_params_optuna


def semantic_dpwc_objective(trial, path, limit):
    n = trial.suggest_categorical('n', [3,5,7])
    latent = trial.suggest_categorical('latent', [True, False])
    k = trial.suggest_categorical('k', [2,3,4,5])
    c = trial.suggest_categorical('c', ['pareto20', 'pareto80', 'knee'])
    kl = trial.suggest_categorical('kl', [3,5,7,10])

    jt = 0.0 #trial.suggest_float('jt', 0.0, 1.0, step=0.01)
    st = 0.15 #trial.suggest_float('st', 0.0, 1.0, step=0.01)

    cutoff = {
        'pareto20': dp.Cutoff.pareto20,
        'pareto80': dp.Cutoff.pareto80,
        'none': dp.Cutoff.none,
        'knee': dp.Cutoff.knee
    }

    semantiMatcher = sm.SemanticMathcer(path=path, limit=limit, model='dpwc', 
    jt=jt, st=st, n=n, latent=latent, k=k, c=cutoff[c], kl=kl)

    relevant = []
    received_cosine = []
    received_jaccard = []
    with open('scenario.json') as json_file:
        scenario = json.load(json_file)
        
        # Train for the service description
        services = scenario['services']
        for s in tqdm.tqdm(services, leave=False):
            semantiMatcher.add(s)

        # Train for the queries
        semantiMatcher.buildIdx()

        # For each test
        tests = ['queries m2m', 'queries one-error', 'queries two-errors-one-word',
        'queries two-errors-two-words', 'queries one-synonym', 'queries two-synonyms',
        'queries three-synonyms', 'queries four-synonyms']

        # Query the system and learn the word profile
        for t in tqdm.tqdm(tests, leave=False):
            queries = scenario[t]
            for q in tqdm.tqdm(queries, leave=False):
                services = semantiMatcher.match(q['query'])

                queryId = int(q['id']) % 100
                relevant.append([queryId])
                received_cosine.append([i for i, _ in services['cosine']])
                received_jaccard.append([i for i, _ in services['jaccard']])
    
    return mm.mean_average_precision(relevant, received_cosine)#,mm.mean_average_precision(relevant, received_jaccard))


def train_semantic_dpwc_matcher(path:str='corpus', limit:int=300):
    # Running the Optuna optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: semantic_dpwc_objective(trial, path, limit), n_trials=100, show_progress_bar=True)

    # Get the best model and hyperparameters
    best_params_optuna = study.best_params

    logger.info(best_params_optuna)
    return best_params_optuna


def main():
    params = train_semantic_dpw_matcher()
    print(f'DPW params: {params}')
    #params = train_semantic_dpwc_matcher()
    #print(f'DPWC params: {params}')
    #semanticMatcher = train_dpw_model('corpus', 300, 3, True, 3, 'knee')
    #print(f'DPW median vector size: {semanticMatcher.model.profile_length}')
    #semanticMatcher = train_dpwc_model('corpus', 300, 3, True, 2, 'knee', 5)
    #print(f'DPWC median vector size: {semanticatcher.model.profile_length}')


if __name__ == '__main__':
    main()
