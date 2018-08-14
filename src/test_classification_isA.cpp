#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

using namespace std;


int dim = 100, sub_test_num = 0, ins_test_num = 0, concept_num = 0, entity_num = 0;
double delta_ins = 0, delta_sub = 0;
bool valid = true;
bool mix = false;
string dataSet = "YAGO39K";

vector<vector<double> > entity_vec, concept_vec;
vector<double> concept_r;
vector<pair<int, int> > ins_right, ins_wrong, sub_right, sub_wrong;

inline double sqr(double x){
    return x * x;
}

bool checkSubClass(int concept1, int concept2){
    double dis = 0;
    for(int i = 0; i < dim; ++i){
        dis += sqr(concept_vec[concept1][i] - concept_vec[concept2][i]);
    }
    if(sqrt(dis) < fabs(concept_r[concept1] - concept_r[concept2]) && concept_r[concept1] < concept_r[concept2]){
        return true;
    }
    if(sqrt(dis) < concept_r[concept1] + concept_r[concept2]){
        double tmp = (concept_r[concept1] + concept_r[concept2] - sqrt(dis)) / concept_r[concept1];
        if(tmp > delta_sub)
            return true;
    }
    return false;
}

bool checkInstance(int instance, int concept){
    double dis = 0;
    for(int i = 0; i < dim; ++i){
        dis += sqr(entity_vec[instance][i] - concept_vec[concept][i]);
    }
    if(sqrt(dis) < concept_r[concept]){
        return true;
    }
    double tmp = concept_r[concept] / sqrt(dis);
    return tmp > delta_ins;
}

void init(){
    entity_vec.clear(); concept_vec.clear();
    concept_r.clear();
    ins_right.clear(); ins_wrong.clear(); sub_right.clear(); sub_wrong.clear();
}

void prepare(){
    init();

    ifstream fin, fin_right;
    if(valid){
        if(mix){
            fin.open(("../data/" + dataSet + "/M-Valid/instanceOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/M-Valid/instanceOf2id_positive.txt").c_str());
        }else{
            fin.open(("../data/" + dataSet + "/Valid/instanceOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/Valid/instanceOf2id_positive.txt").c_str());
        }
    }else{
        if(mix){
            fin.open(("../data/" + dataSet + "/M-Test/instanceOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/M-Test/instanceOf2id_positive.txt").c_str());
        }else{
            fin.open(("../data/" + dataSet + "/Test/instanceOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/Test/instanceOf2id_positive.txt").c_str());
        }
    }
    fin >> ins_test_num;
    fin_right >> ins_test_num;
    int tmp1, tmp2;
    for(int i = 0; i < ins_test_num; ++i){
        fin >> tmp1 >> tmp2;
        ins_wrong.emplace_back(tmp1, tmp2);
        fin_right >> tmp1 >> tmp2;
        ins_right.emplace_back(tmp1, tmp2);
    }
    fin.close();
    fin_right.close();
    if(valid){
        if(mix){
            fin.open(("../data/" + dataSet + "/M-Valid/subClassOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/M-Valid/subClassOf2id_positive.txt").c_str());
        }else{
            fin.open(("../data/" + dataSet + "/Valid/subClassOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/Valid/subClassOf2id_positive.txt").c_str());
        }
    }else {
        if(mix){
            fin.open(("../data/" + dataSet + "/M-Test/subClassOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/M-Test/subClassOf2id_positive.txt").c_str());
        }else{
            fin.open(("../data/" + dataSet + "/Test/subClassOf2id_negative.txt").c_str());
            fin_right.open(("../data/" + dataSet + "/Test/subClassOf2id_positive.txt").c_str());
        }
    }
    fin >> sub_test_num;
    fin_right >> sub_test_num;
    for(int i = 0; i < sub_test_num; ++i){
        fin >> tmp1 >> tmp2;
        sub_wrong.emplace_back(tmp1, tmp2);
        fin_right >> tmp1 >> tmp2;
        sub_right.emplace_back(tmp1, tmp2);
    }
    fin.close();
    fin_right.close();

    int tmp = 0;
    FILE *fin_num = fopen(("../data/" + dataSet + "/Train/instance2id.txt").c_str(), "r");
    tmp = fscanf(fin_num, "%d", &entity_num);
    fclose(fin_num);
    fin_num = fopen(("../data/" + dataSet + "/Train/concept2id.txt").c_str(), "r");
    tmp = fscanf(fin_num, "%d", &concept_num);
    fclose(fin_num);

    FILE* f1 = fopen(("vector/" + dataSet + "/entity2vec.vec").c_str(), "r");
    FILE* f2 = fopen(("vector/" + dataSet + "/concept2vec.vec").c_str(), "r");
    entity_vec.resize(entity_num);
    for(int i = 0; i < entity_num; ++i){
        entity_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            fscanf(f1, "%lf", &entity_vec[i][j]);
        }
    }
    concept_vec.resize(concept_num);
    concept_r.resize(concept_num);
    for(int i = 0; i < concept_num; ++i){
        concept_vec[i].resize(dim);
        for(int j = 0; j < dim; ++j){
            fscanf(f2, "%lf", &concept_vec[i][j]);
        }
        fscanf(f2, "%lf", &concept_r[i]);
    }
}

pair<double, double> test(){
    double TP_ins = 0, TN_ins = 0, FP_ins = 0, FN_ins = 0;
    double TP_sub = 0, TN_sub = 0, FP_sub = 0, FN_sub = 0;
    map<int, double> TP_ins_map, TN_ins_map, FP_ins_map, FN_ins_map;
    set<double> concept_set;

    for(int i = 0; i < ins_test_num; ++i){
        if(checkInstance(ins_right[i].first, ins_right[i].second)) {
            TP_ins++;
            if(TP_ins_map.count(ins_right[i].second) > 0){
                TP_ins_map[ins_right[i].second]++;
            }else{
                TP_ins_map[ins_right[i].second] = 1;
            }
        }else {
            FN_ins++;
            if(FN_ins_map.count(ins_right[i].second) > 0){
                FN_ins_map[ins_right[i].second]++;
            }else{
                FN_ins_map[ins_right[i].second] = 1;
            }
        }
        if(!checkInstance(ins_wrong[i].first, ins_wrong[i].second)) {
            TN_ins++;
            if(TN_ins_map.count(ins_wrong[i].second) > 0){
                TN_ins_map[ins_wrong[i].second]++;
            }else{
                TN_ins_map[ins_wrong[i].second] = 1;
            }
        }else {
            FP_ins++;
            if(FP_ins_map.count(ins_wrong[i].second) > 0){
                FP_ins_map[ins_wrong[i].second]++;
            }else{
                FP_ins_map[ins_wrong[i].second] = 1;
            }
        }
        double concept_s = ins_right[i].second;
        double concept_m = ins_wrong[i].second;
        concept_set.insert(concept_s);
        concept_set.insert(concept_m);
    }
    for(int i = 0; i < sub_test_num; ++i){
        if(checkSubClass(sub_right[i].first, sub_right[i].second))
            TP_sub++;
        else
            FN_sub++;
        if(!checkSubClass(sub_wrong[i].first, sub_wrong[i].second))
            TN_sub++;
        else
            FP_sub++;
    }
    if(valid){
        double ins_ans = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins);
        double sub_ins = (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub);
        return make_pair(ins_ans, sub_ins);
    }else{
        cout << "instanceOf triple classification:" << endl;
        cout << "accuracy: " << (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins) << "%" << endl;
        cout << "precision: " << TP_ins * 100 /(TP_ins + FP_ins) << "%" << endl;
        cout << "recall: " << TP_ins * 100 / (TP_ins + FN_ins) << "%" << endl;
        double p = TP_ins * 100 /(TP_ins + FP_ins), r = TP_ins * 100 / (TP_ins + FN_ins);
        cout << "F1-score: " << 2 * p * r / (p + r) << "%" << endl;
        cout << endl;
        cout << "subClassOf triple classification:" << endl;
        cout << "accuracy: " << (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub) << "%" << endl;
        cout << "precision: " << TP_sub * 100 /(TP_sub + FP_sub) << "%" << endl;
        cout << "recall: " << TP_sub * 100 / (TP_sub + FN_sub) << "%" << endl;
        p = TP_sub * 100 /(TP_sub + FP_sub), r = TP_sub * 100 / (TP_sub + FN_sub);
        cout << "F1-score: " << 2 * p * r / (p + r) << "%" << endl;

        for(set<double>::iterator iter = concept_set.begin(); iter != concept_set.end(); ++iter){
            int index = *iter;
            TP_ins = TP_ins_map[index];
            TN_ins = TN_ins_map[index];
            FN_ins = FN_ins_map[index];
            FP_ins = FP_ins_map[index];
            double accuracy = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins);
            double precision = TP_ins * 100 /(TP_ins + FP_ins);
            double recall = TP_ins * 100 / (TP_ins + FN_ins);
            p = TP_ins * 100 /(TP_ins + FP_ins);
            r = TP_ins * 100 / (TP_ins + FN_ins);
            double f1 = 2 * p * r / (p + r);
        }
        return make_pair(0.0, 0.0);
    }
}

void runValid(){
    double ins_best_answer = 0, ins_best_delta = 0;
    double sub_best_answer = 0, sub_best_delta = 0;
    for(int i = 0; i < 101; ++i){
        double f = i; f /= 100;
        delta_ins = f;
        delta_sub = f * 2;
        pair<double, double> ans = test();
        if(ans.first > ins_best_answer){
            ins_best_answer = ans.first;
            ins_best_delta = f;
        }
        if(ans.second > sub_best_answer){
            sub_best_answer = ans.second;
            sub_best_delta = f * 2;
        }
    }
    cout << "delta_ins is " << ins_best_delta << ". The best ins accuracy on valid data is " << ins_best_answer << "%" << endl;
    cout << "delta_sub is " << sub_best_delta << ". The best sub accuracy on valid data is " << sub_best_answer << "%" << endl;
    cout << endl;
    delta_ins = ins_best_delta;
    delta_sub = sub_best_delta;
    valid = false;
    prepare();
    test();
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char**argv){
    int i = 0;
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) dataSet = argv[i + 1];
    if ((i = ArgPos((char *)"-mix", argc, argv)) > 0) mix = static_cast<bool>(atoi(argv[i + 1]));
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    cout << "data = " << dataSet << endl;
    if (mix)
        cout << "mix = " << "True" << endl;
    else
        cout << "mix = " << "False" << endl;
    cout << "dimension = " << dim << endl;
    prepare();
    runValid();
}
