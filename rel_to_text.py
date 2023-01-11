from sys import argv

#### IDEA:
# - Generate mediocre sentences
# - Train model on MLM but also on sentence fixing (randomly corrupt
#   sentences)
# - Train model on RMLM (Relation-MLM), but perform masking on prediction
#   of what the model thinks the sentence should be.

replace = {
    'may_treat': "{} may help with the treatment of {}.\n",
    'may_prevent': "{} may help prevent {}.\n",
    'cause_of': "{} can cause {}.\n",
    'causative_agent_of': "{} is a risk factor for {}.\n",
    'contraindicated_with_disease':
        "Use of {} is discouraged in individuals affected by {}.\n",
    'isa': "{} is a type of {}.\n",
    'associated_with': "{} is associated with {}.\n",
    'clinically_associated_with': "{} is clinically associated with {}.\n",
    'co-occurs_with': "{} can co-occur with {}.\n",
    'has_method': "{} is an application of {}.\n",
    'tradename_of': "{} is a brand name for {}.\n",
    'measures': "{} can be used to measure {} levels\n",
    # 'measures': "CUI1 levels can be measured using a CUI2\n",
    'part_of': "The {} is a part of the {}.\n",
    'member_of': "{} is classified under {}.\n",
    'finding_method_of': "{} can be used to identify {}.\n",
    'possibly_equivalent_to':
        'The concept of "{}" may be equivalent to {}.\n',
    'same_as': 'The concept of "{}" and "{}" are equivalent.\n',
    'active_ingredient_of': '{} is an active ingredient in {}.\n',
    'inactive_ingredient_of': '{} is an inactive ingredient in {}.\n',
    'concept_in_subset': 'The concept of "{}" is part of the {}.\n',
    "has_manifestation": '{} may manifest as {}.\n',
    "ingredient_of": '{} is an ingredient in {}.\n',
    "classifies": 'The "{}" category includes "{}".\n',
    "mapped_to": '{} is an example of {}.\n',
    "consists_of": '{} consists of {}.\n',
    "is_associated_anatomic_site_of":
        'The {} is the anatomic site associated with {}.\n',
    "gene_plays_role_in_process":
        'The {} plays a role in the process of {}.\n',
    "occurs_in": '{} only occurs during {}.\n'
}


if __name__ == '__main__':
    try:
        in_fname = argv[1]
        out_fname = argv[2]
    except IndexError:
        print("Missing filename. Usage:\n"
              "$ python rel_to_text.py input_file output_file")
        exit(1)
    with open(in_fname, 'r') as f:
        # [-2:0:-1] removes the first and last columns (CUIs) and saves
        # relations backwards (they are already stored backwards in the
        # table).
        relations = [rel[-2:0:-1] for l in f.readlines()
                     if len(rel := l.strip().split('\t')) == 5]

    # header line
    del relations[0]

    for i, rel in enumerate(relations):
        relations[i] = replace[rel[1]].format(rel[0], rel[2])

    with open(out_fname, 'w') as f:
        f.writelines(relations)
