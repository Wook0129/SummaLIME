import numpy as np
from nltk.corpus import stopwords


stoplist = set(stopwords.words('english'))


def get_explanation(explainer, pipeline, instance):
    
    #  Input : (LIME Text Explainer) explainer, (String) Text to Explain Result
    #  Output : (Dictionary) explanation (key, value = term, coef), (int) label of text
    
    explanation = {}
    num_sentence = len(instance.split('. '))
    exp = explainer.explain_instance(instance, pipeline.predict_proba, num_features=num_sentence * 2, num_samples=300)
    label = np.argmax(exp.predict_proba)
    
    for term_coef in exp.as_list():
            
        term = term_coef[0].lower()
        coef = term_coef[1]
        if term not in stoplist:
            explanation[term] = coef

    return [explanation, instance, label, exp]


def get_feature_importances(explanations):

    #  Input : (List of LIME Text Explainer) explanations
    #  Output : (Dictionary) feature importances (key, value = term, coef)

    feature_importances = {}
    
    for explanation in explanations:
        
        term_coefs = explanation[0]
        
        for term in term_coefs.keys():
            
            coef = term_coefs[term]

            if term in feature_importances.keys():
                feature_importances[term] += np.abs(coef)
            else:
                feature_importances[term] = np.abs(coef)

    return feature_importances


def submodular_pick(explanations, feature_importances, l_penalty = False, k=20):
    
    # Input
    # (List of LIME Text Explainer) explanations
    # (Dictionary) feature importances(key, value = term, coef)

    # Output
    # (List of LIME Text Explainer) Selected Explanations, (Set) Used Features

    chosen = []

    temp = explanations.copy()
    
    features = feature_importances.keys()
    used_features = set()
    
    while len(chosen) != k:
        
        best_gain = 0
        best_item = temp[0]
        
        best_newly_added_features = set()
        
        for explanation in temp:
            
            original_text = explanation[1]
            
            text_length = len(original_text.split())
            length_penalty = np.sqrt(text_length)
            
            term_coefs = explanation[0]
            terms = set(term_coefs.keys())
            
            newly_added_features = set()
            
            feature_importance_sum = 0
                
            for term in terms:
                if (term not in stoplist) and (term in features):
                    feature_importance_sum += np.sqrt(feature_importances[term])
                    newly_added_features.union(term)
            
            gain = feature_importance_sum
            
            if l_penalty == True:
                gain /= length_penalty

            if gain > best_gain:
                best_gain = gain
                best_item = explanation
                best_newly_added_features = newly_added_features
                
        used_features.union(best_newly_added_features)
        
        chosen.append([best_item, best_newly_added_features])
        temp.remove(best_item)

    return chosen, used_features
