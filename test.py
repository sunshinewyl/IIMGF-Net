import numpy as np
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from src.eval_metrics import ConfusionMatrix
import yaml
from types import SimpleNamespace
import argparse
import torch
from torch.utils.data import DataLoader
import os
from models.IIMGF import IIMGF
from src.dataloader import load_dataset,dataset

'''function for saving model'''
def modelSnapShot(model,newModelPath,oldModelPath=None,onlyBestModel=False):
    if onlyBestModel and oldModelPath:
        os.remove(oldModelPath)
    torch.save(model.state_dict(),newModelPath)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument('--config', type=str, default='/home/wyl/work/Rebuttal/IIMGF_yaml/config/test.yaml',
                        required=True, help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # 将 dict 转成可以用 . 访问的形式，例如 config.train.epochs
    return SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config_dict.items()})

def main():
    args = parse_args()
    config = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.model.gpu_idx
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IIMGF(config=config).to(device)

    # load dataset
    derm_data_group = load_dataset(dir_release=config.data.derm7pt_path)
    test_iterator = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='test'),
                               batch_size=1, shuffle=False, num_workers=2)


    log_path = config.output.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    label_t = [0,1,2,3,4]
    class_num = 5
    log_file = open(log_path + 'test_log.txt', 'w')

    print("Start testing...")
    log_file.write("Start testing...")
    log_file.flush()

    y_true_diag = []
    y_true_pn = []
    y_true_bmv = []
    y_true_vs = []
    y_true_pig = []
    y_true_str = []
    y_true_dag = []
    y_true_rs = []
    y_score_diag = []
    y_score_pn = []
    y_score_bmv = []
    y_score_vs = []
    y_score_pig = []
    y_score_str = []
    y_score_dag = []
    y_score_rs = []


    try:

        model.load_state_dict(torch.load(config.output.model_path), strict=True)

        model.eval()

        avg_test_loss = 0
        confusion_diag =ConfusionMatrix(num_classes=class_num,labels =label_t)
        confusion_pn = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_bmv = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_vs = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_pig = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_str = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_dag = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_rs = ConfusionMatrix(num_classes=class_num, labels=label_t)

        for der_data,cli_data,meta_data,target in test_iterator:
            # target=torch.squeeze(target,1) #torch.squeeze()对数据维数进行压缩，去掉target中维数为1的维度
            # Diagostic label
            diagnosis_label = target[0].squeeze(1).cuda()
            # Seven-Point Checklikst labels
            pn_label = target[1].squeeze(1).cuda()
            bmv_label = target[2].squeeze(1).cuda()
            vs_label = target[3].squeeze(1).cuda()
            pig_label = target[4].squeeze(1).cuda()
            str_label = target[5].squeeze(1).cuda()
            dag_label = target[6].squeeze(1).cuda()
            rs_label = target[7].squeeze(1).cuda()

            der_data,cli_data,meta_data= der_data.cuda(), cli_data.cuda(),meta_data.cuda().float()
            der_data,cli_data,meta_data= Variable(der_data),Variable(cli_data),Variable(meta_data)


            output = model(cli_data,der_data)

            prob_diag = torch.softmax(output[0],dim=1)
            prob_pn = torch.softmax(output[1],dim=1)
            prob_bmv = torch.softmax(output[2],dim=1)
            prob_vs = torch.softmax(output[3],dim=1)
            prob_pig = torch.softmax(output[4],dim=1)
            prob_str = torch.softmax(output[5],dim=1)
            prob_dag = torch.softmax(output[6],dim=1)
            prob_rs = torch.softmax(output[7],dim=1)
            y_true_diag.append(diagnosis_label.cpu().numpy())
            y_true_pn.append(pn_label.cpu().numpy())
            y_true_bmv.append(bmv_label.cpu().numpy())
            y_true_vs.append(vs_label.cpu().numpy())
            y_true_pig.append(pig_label.cpu().numpy())
            y_true_str.append(str_label.cpu().numpy())
            y_true_dag.append(dag_label.cpu().numpy())
            y_true_rs.append(rs_label.cpu().numpy())
            y_score_diag.append(prob_diag.cpu().detach().numpy())
            y_score_pn.append(prob_pn.cpu().detach().numpy())
            y_score_bmv.append(prob_bmv.cpu().detach().numpy())
            y_score_vs.append(prob_vs.cpu().detach().numpy())
            y_score_pig.append(prob_pig.cpu().detach().numpy())
            y_score_str.append(prob_str.cpu().detach().numpy())
            y_score_dag.append(prob_dag.cpu().detach().numpy())
            y_score_rs.append(prob_rs.cpu().detach().numpy())

            test_loss = torch.true_divide(
                model.criterion(output[0],diagnosis_label)
                + model.criterion(output[1],pn_label)
                + model.criterion(output[2],bmv_label)
                + model.criterion(output[3],vs_label)
                + model.criterion(output[4],pig_label)
                + model.criterion(output[5],str_label)
                + model.criterion(output[6],dag_label)
                + model.criterion(output[7],rs_label),8
            )

            #confusion matrix
            ret,predictions_diag = torch.max(output[0].data,1)
            ret, predictions_pn = torch.max(output[1].data, 1)
            ret, predictions_bmv = torch.max(output[2].data, 1)
            ret, predictions_vs = torch.max(output[3].data, 1)
            ret, predictions_pig = torch.max(output[4].data, 1)
            ret, predictions_str = torch.max(output[5].data, 1)
            ret, predictions_dag = torch.max(output[6].data, 1)
            ret, predictions_rs = torch.max(output[7].data, 1)

            confusion_diag.update(predictions_diag.cpu().numpy(),diagnosis_label.cpu().numpy())
            confusion_pn.update(predictions_pn.cpu().numpy(), pn_label.cpu().numpy())
            confusion_bmv.update(predictions_bmv.cpu().numpy(), bmv_label.cpu().numpy())
            confusion_vs.update(predictions_vs.cpu().numpy(), vs_label.cpu().numpy())
            confusion_pig.update(predictions_pig.cpu().numpy(), pig_label.cpu().numpy())
            confusion_str.update(predictions_str.cpu().numpy(), str_label.cpu().numpy())
            confusion_dag.update(predictions_dag.cpu().numpy(), dag_label.cpu().numpy())
            confusion_rs.update(predictions_rs.cpu().numpy(), rs_label.cpu().numpy())

        diag_auc = roc_auc_score(np.concatenate(y_true_diag, axis=0), np.concatenate(y_score_diag, axis=0), multi_class='ovr', average='macro')
        pn_auc = roc_auc_score(np.concatenate(y_true_pn, axis=0), np.concatenate(y_score_pn, axis=0), multi_class='ovr', average='macro')
        bmv_auc = roc_auc_score(np.concatenate(y_true_bmv, axis=0), np.concatenate(y_score_bmv, axis=0)[:,1])
        vs_auc = roc_auc_score(np.concatenate(y_true_vs, axis=0), np.concatenate(y_score_vs, axis=0),
                                 multi_class='ovr', average='macro')
        pog_auc = roc_auc_score(np.concatenate(y_true_pig, axis=0), np.concatenate(y_score_pig, axis=0), multi_class='ovr',
                               average='macro')
        str_auc = roc_auc_score(np.concatenate(y_true_str, axis=0), np.concatenate(y_score_str, axis=0),
                                multi_class='ovr', average='macro')
        dag_auc = roc_auc_score(np.concatenate(y_true_dag, axis=0), np.concatenate(y_score_dag, axis=0),
                                 multi_class='ovr', average='macro')
        rs_auc = roc_auc_score(np.concatenate(y_true_rs, axis=0), np.concatenate(y_score_rs, axis=0)[:,1])

        print('diag:',diag_auc)
        print('pn:',pn_auc)
        print('bmv:',bmv_auc)
        print('vs:',vs_auc)
        print('pig:',pog_auc)
        print('str:',str_auc)
        print('dag:',dag_auc)
        print('rs:',rs_auc)
        print('avg', (diag_auc+pn_auc+bmv_auc+vs_auc+pog_auc+str_auc+dag_auc+rs_auc)/8)



        print("Daig:\n")
        log_file.write("Daig:\n")
        diag_acc = confusion_diag.summary(log_file)
        print("PN:\n")
        log_file.write("PN:\n")
        PN_acc = confusion_pn.summary(log_file)
        print("BMV:\n")
        log_file.write("BMV:\n")
        bmv_acc = confusion_bmv.summary(log_file)
        print("VS:\n")
        log_file.write("VS:\n")
        vs_acc = confusion_vs.summary(log_file)
        print("PIG:\n")
        log_file.write("PIG:\n")
        pig_acc = confusion_pig.summary(log_file)
        print("STR:\n")
        log_file.write("STR:\n")
        str_acc = confusion_str.summary(log_file)
        print("DAG:\n")
        log_file.write("DAG:\n")
        dag_acc = confusion_dag.summary(log_file)
        print("RS:\n")
        log_file.write("RS:\n")
        rs_acc = confusion_rs.summary(log_file)
        log_file.flush()
        avg_acc = (diag_acc + PN_acc + bmv_acc + vs_acc + pig_acc + str_acc + dag_acc + rs_acc) / 8.0
        print('avg_acc=', avg_acc)

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        log_file.close()


if __name__ == "__main__":
    main()
