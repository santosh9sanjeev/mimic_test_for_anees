# # import os
# # import pickle
# # import pprint
# # import random
# # from glob import glob
# # from os.path import exists, join
# # from tqdm import tqdm
# # import numpy as np
# # import torch
# # import sklearn.metrics
# # from sklearn.metrics import roc_auc_score, accuracy_score
# # import sklearn, sklearn.model_selection



# # def train(model, dataset, train_loader, valid_loader, device, args):    

# #     print(args.output_dir)

# #     if not exists(args.output_dir):
# #         os.makedirs(args.output_dir)

# #     dataset_name = 'MIMIC-densenet'
# #     # Optimizer
# #     optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
# #     print(optim)

# #     criterion = torch.nn.BCEWithLogitsLoss()

# #     # Checkpointing
# #     start_epoch = 0
# #     best_metric = 0.
# #     weights_for_best_validauc = None
# #     auc_test = None
# #     metrics = []
# #     weights_files = glob(join(args.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
# #     if len(weights_files):
# #         # Find most recent epoch
# #         epochs = np.array(
# #             [int(w[len(join(args.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
# #         start_epoch = epochs.max()
# #         weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
# #         model.load_state_dict(torch.load(weights_file).state_dict())

# #         with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
# #             metrics = pickle.load(f)

# #         best_metric = metrics[-1]['best_metric']
# #         weights_for_best_validauc = model.state_dict()

# #         print("Resuming training at epoch {0}.".format(start_epoch))
# #         print("Weights loaded: {0}".format(weights_file))

# #     model.to(device)
    
# #     for epoch in range(start_epoch, args.n_epochs):

# #         avg_loss = train_epoch(cfg=args,
# #                                epoch=epoch,
# #                                model=model,
# #                                device=device,
# #                                optimizer=optim,
# #                                train_loader=train_loader,
# #                                criterion=criterion)
        
# #         auc_valid = valid_test_epoch(name='Valid',
# #                                      epoch=epoch,
# #                                      model=model,
# #                                      device=device,
# #                                      data_loader=valid_loader,
# #                                      criterion=criterion)[0]

# #         if np.mean(auc_valid) > best_metric:
# #             best_metric = np.mean(auc_valid)
# #             weights_for_best_validauc = model.state_dict()
# #             torch.save(model, join(args.output_dir, f'{dataset_name}-best.pt'))
# #             # only compute when we need to

# #         stat = {
# #             "epoch": epoch + 1,
# #             "trainloss": avg_loss,
# #             "validauc": auc_valid,
# #             'best_metric': best_metric
# #         }

# #         metrics.append(stat)

# #         with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
# #             pickle.dump(metrics, f)

# #         torch.save(model, join(args.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

# #     return metrics, best_metric, weights_for_best_validauc





# # def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, limit=None):
# #     model.train()

# #     if cfg.taskweights:
# #         weights = np.nansum(train_loader.dataset.labels, axis=0)
# #         weights = weights.max() - weights + weights.mean()
# #         weights = weights/weights.max()
# #         weights = torch.from_numpy(weights).to(device).float()
# #         print("task weights", weights)
    
# #     avg_loss = []
# #     t = tqdm(train_loader)
# #     for batch_idx, samples in enumerate(t):
        
# #         if limit and (batch_idx > limit):
# #             print("breaking out")
# #             break
            
# #         optimizer.zero_grad()
        
# #         images = samples["img"].float().to(device)
# #         targets = samples["lab"].to(device)
# #         outputs = model(images)
        
# #         loss = torch.zeros(1).to(device).float()
# #         for task in range(targets.shape[1]):
# #             task_output = outputs[:,task]
# #             task_target = targets[:,task]
# #             mask = ~torch.isnan(task_target)
# #             task_output = task_output[mask]
# #             task_target = task_target[mask]
# #             if len(task_target) > 0:
# #                 task_loss = criterion(task_output.float(), task_target.float())
# #                 if cfg.taskweights:
# #                     loss += weights[task]*task_loss
# #                 else:
# #                     loss += task_loss
        
# #         # here regularize the weight matrix when label_concat is used
# #         if cfg.label_concat_reg:
# #             if not cfg.label_concat:
# #                 raise Exception("cfg.label_concat must be true")
# #             weight = model.classifier.weight
# #             num_labels = 13
# #             num_datasets = weight.shape[0]//num_labels
# #             weight_stacked = weight.reshape(num_datasets,num_labels,-1)
# #             label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
# #             for task in range(num_labels):
# #                 dists = torch.pdist(weight_stacked[:,task], p=2).mean()
# #                 loss += label_concat_reg_lambda*dists
                
# #         loss = loss.sum()
        
# #         if cfg.featurereg:
# #             feat = model.features(images)
# #             loss += feat.abs().sum()
            
# #         if cfg.weightreg:
# #             loss += model.classifier.weight.abs().sum()
        
# #         loss.backward()

# #         avg_loss.append(loss.detach().cpu().numpy())
# #         t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

# #         optimizer.step()

# #     return np.mean(avg_loss)

# # def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
# #     model.eval()

# #     avg_loss = []
# #     task_outputs={}
# #     task_targets={}
# #     for task in range(data_loader.dataset[0]["lab"].shape[0]):
# #         task_outputs[task] = []
# #         task_targets[task] = []
        
# #     with torch.no_grad():
# #         t = tqdm(data_loader)
# #         for batch_idx, samples in enumerate(t):

# #             if limit and (batch_idx > limit):
# #                 print("breaking out")
# #                 break
            
# #             images = samples["img"].to(device)
# #             targets = samples["lab"].to(device)

# #             outputs = model(images)
            
# #             loss = torch.zeros(1).to(device).double()
# #             for task in range(targets.shape[1]):
# #                 task_output = outputs[:,task]
# #                 task_target = targets[:,task]
# #                 mask = ~torch.isnan(task_target)
# #                 task_output = task_output[mask]
# #                 task_target = task_target[mask]
# #                 if len(task_target) > 0:
# #                     loss += criterion(task_output.double(), task_target.double())
                
# #                 task_outputs[task].append(task_output.detach().cpu().numpy())
# #                 task_targets[task].append(task_target.detach().cpu().numpy())

# #             loss = loss.sum()
            
# #             avg_loss.append(loss.detach().cpu().numpy())
# #             t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
# #         for task in range(len(task_targets)):
# #             task_outputs[task] = np.concatenate(task_outputs[task])
# #             task_targets[task] = np.concatenate(task_targets[task])
    
# #         task_aucs = []
# #         for task in range(len(task_targets)):
# #             if len(np.unique(task_targets[task]))> 1:
# #                 task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
# #                 #print(task, task_auc)
# #                 task_aucs.append(task_auc)
# #             else:
# #                 task_aucs.append(np.nan)

# #     task_aucs = np.asarray(task_aucs)
# #     auc = np.mean(task_aucs[~np.isnan(task_aucs)])
# #     print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

# #     return auc, task_aucs, task_outputs, task_targets


# # def valid_test_epoch_v2(name, dataset, model, device, data_loader, criterion, healthy, limit=None):
# #     model.eval()

# #     avg_loss = []
# #     task_outputs={}
# #     task_targets={}
# #     for task in range(13):
# #         task_outputs[task] = []
# #         task_targets[task] = []
        
# #     with torch.no_grad():
# #         t = tqdm(data_loader)
# #         for batch_idx, samples in enumerate(t):

# #             if limit and (batch_idx > limit):
# #                 print("breaking out")
# #                 break
            
# #             images = samples["img"].to(device)
# #             targets = samples["lab"].to(device)

# #             outputs = model(images)
            
# #             loss = torch.zeros(1).to(device).double()
# #             for task in range(targets.shape[1]):
# #                 task_output = outputs[:,task]
# #                 task_target = targets[:,task]
# #                 mask = ~torch.isnan(task_target)
# #                 task_output = task_output[mask]
# #                 task_target = task_target[mask]
# #                 if len(task_target) > 0:
# #                     loss += criterion(task_output.double(), task_target.double())
                
# #                 task_outputs[task].append(task_output.detach().cpu().numpy())
# #                 task_targets[task].append(task_target.detach().cpu().numpy())

# #             loss = loss.sum()
            
# #             avg_loss.append(loss.detach().cpu().numpy())
# #             t.set_description(f'Loss = {np.mean(avg_loss):4.4f}')
            
# #         for task in range(len(task_targets)):
# #             task_outputs[task] = np.concatenate(task_outputs[task])
# #             task_targets[task] = np.concatenate(task_targets[task])
    
# #         task_aucs = []
# #         for task in range(len(task_targets)):
# #             if len(np.unique(task_targets[task]))> 1:
# #                 task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
# #                 #print(task, task_auc)
# #                 task_aucs.append(task_auc)
# #             else:
# #                 task_aucs.append(np.nan)

# #     task_aucs = np.asarray(task_aucs)
# #     auc = np.mean(task_aucs[~np.isnan(task_aucs)])
# #     print(f'Avg AUC = {auc:4.4f}')
    
# #     results = [auc, task_aucs, task_outputs, task_targets]
# #     # all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]

# #     if healthy == True :
# #         print('hiiiiiiiii')
# #         # [0.5938055, 0.5748906, 0.30114534, 0.18698719, 0.12614557, 0.04757202, 0.092572555, 0.5241535, 0.63551843, 0.028975938, 0.22440709, 0.09897159, 0.6400715]
# #         all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]
# #         all_min = []
# #         all_max = []
# #         all_ppv80 = []
# #         all_accuracy = []
# #         all_f1_score = []
# #         all_precision = []
# #         all_recall = []
# #         all_auc = []
# #         ls = []
# #         # print(results)
# #         for i, patho in enumerate(dataset.pathologies):
# #             print(i, patho)
# #             opt_thres = np.nan
# #             opt_min = np.nan
# #             opt_max = np.nan
# #             ppv80_thres = np.nan
# #             accuracy = np.nan
# #             f1_score = np.nan
# #             precision = np.nan
# #             recall = np.nan
# #             auc = np.nan
            
# #             # if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
                
# #             #sigmoid
# #             all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
            
# #             # fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
# #             # pente = tpr - fpr
# #             # opt_thres = thres_roc[np.argmax(pente)]
# #             # opt_min = all_outputs.min()
# #             # opt_max = all_outputs.max()
            
# #             # ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
# #             # ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
# #             # ppv80_thres = thres_pr[ppv80_thres_idx-1]
            
# #             # auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
            
# #             # Calculate confusion matrix for accuracy, precision, recall, and F1 score
# #             threshold = all_threshs[i]  
# #             predicted_labels = (all_outputs >= threshold).astype(int)
# #             true_labels = results[3][i]
# #             print(patho)
# #             print(predicted_labels)
# #             print(true_labels)
# #             ls.append(predicted_labels)
# #         array = np.array(ls)
# #         arr2 = array.T
# #         print(arr2, arr2.shape)

# #         rows_with_all_zeros = np.sum(np.all(arr2==0,axis=1))
# #         print(rows_with_all_zeros)
            

# #     else:
# #         perf_dict = {}
# #         # [0.5938055, 0.5748906, 0.30114534, 0.18698719, 0.12614557, 0.04757202, 0.092572555, 0.5241535, 0.63551843, 0.028975938, 0.22440709, 0.09897159, 0.6400715]
# #         all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]
# #         all_min = []
# #         all_max = []
# #         all_ppv80 = []
# #         all_accuracy = []
# #         all_f1_score = []
# #         all_precision = []
# #         all_recall = []
# #         all_auc = []
# #         # print(results)
# #         for i, patho in enumerate(dataset.pathologies):
# #             print(i, patho)
# #             opt_thres = np.nan
# #             opt_min = np.nan
# #             opt_max = np.nan
# #             ppv80_thres = np.nan
# #             accuracy = np.nan
# #             f1_score = np.nan
# #             precision = np.nan
# #             recall = np.nan
# #             auc = np.nan
            
# #             if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
                
# #                 #sigmoid
# #                 all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
                
# #                 fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
# #                 pente = tpr - fpr
# #                 opt_thres = thres_roc[np.argmax(pente)]
# #                 opt_min = all_outputs.min()
# #                 opt_max = all_outputs.max()
                
# #                 ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
# #                 ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
# #                 ppv80_thres = thres_pr[ppv80_thres_idx-1]
                
# #                 auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
                
# #                 # Calculate confusion matrix for accuracy, precision, recall, and F1 score
# #                 threshold = all_threshs[i]  
# #                 predicted_labels = (all_outputs >= threshold).astype(int)
# #                 true_labels = results[3][i]
# #                 confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
# #                 TP = confusion_matrix[1, 1]
# #                 TN = confusion_matrix[0, 0]
# #                 FP = confusion_matrix[0, 1]
# #                 FN = confusion_matrix[1, 0]

# #                 # Calculate metrics
# #                 accuracy = (TP + TN) / (TP + TN + FP + FN)
# #                 precision = TP / (TP + FP)
# #                 recall = TP / (TP + FN)
# #                 f1_score = 2 * (precision * recall) / (precision + recall)
                
# #                 # Add metrics to perf_dict
# #                 perf_dict[patho] = {
# #                     'AUC': round(auc, 2),
# #                     'Accuracy': round(accuracy, 2),
# #                     'F1 Score': round(f1_score, 2),
# #                     'Precision': round(precision, 2),
# #                     'Recall': round(recall, 2)
# #                 }
                
# #                 all_auc.append(auc)  # Append AUC to the list
                
# #             else:
# #                 perf_dict[patho] = "-"
        



# #             # Append metrics to respective lists
# #             all_threshs.append(opt_thres)
# #             all_min.append(opt_min)
# #             all_max.append(opt_max)
# #             all_ppv80.append(ppv80_thres)
# #             all_accuracy.append(accuracy)
# #             all_f1_score.append(f1_score)
# #             all_precision.append(precision)
# #             all_recall.append(recall)


# #         # for i in enumerate(len(task_targets)):
# #         #     c=0
# #         #     for  j,patho in enumerate(dataset.pathologies):
# #         #         if task_targets[i][j] != 0:
# #         #             c=1
# #         #             break
# #         #     if c==0:
                

# #             print(i, patho)
# #             opt_thres = np.nan
# #             opt_min = np.nan
# #             opt_max = np.nan
# #             ppv80_thres = np.nan
# #             accuracy = np.nan
# #             f1_score = np.nan
# #             precision = np.nan
# #             recall = np.nan
# #             auc = np.nan
            
# #             # if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):





# #         # Print the results
# #         print("pathologies", dataset.pathologies)
# #         print("------------------------------------------------------------------------------------------------")
# #         print("op_threshs", str(all_threshs).replace("nan", "np.nan"))
# #         print("min", str(all_min).replace("nan", "np.nan"))
# #         print("max", str(all_max).replace("nan", "np.nan"))
# #         print("ppv80", str(all_ppv80).replace("nan", "np.nan"))
# #         print("accuracy", str(all_accuracy).replace("nan", "np.nan"))
# #         print("f1_score", str(all_f1_score).replace("nan", "np.nan"))
# #         print("precision", str(all_precision).replace("nan", "np.nan"))
# #         print("recall", str(all_recall).replace("nan", "np.nan"))
# #         print("all AUC values:", str(all_auc).replace("nan", "np.nan"))

# #         # Calculate and print average metrics
# #         avg_accuracy = np.nanmean(all_accuracy)
# #         avg_f1_score = np.nanmean(all_f1_score)
# #         avg_precision = np.nanmean(all_precision)
# #         avg_recall = np.nanmean(all_recall)
# #         avg_auc = np.nanmean(all_auc)

# #         print(f'Average Accuracy: {round(avg_accuracy, 2)}')
# #         print(f'Average F1 Score: {round(avg_f1_score, 2)}')
# #         print(f'Average Precision: {round(avg_precision, 2)}')
# #         print(f'Average Recall: {round(avg_recall, 2)}')
# #         print(f'Average AUC: {round(avg_auc, 2)}')
            





# # def validate(model, checkpoint_path, dataset, valid_loader, device, args):
# #     print('inside validate',len(dataset), valid_loader)
# #     model.load_state_dict(torch.load(checkpoint_path).state_dict())
# #     criterion = torch.nn.BCEWithLogitsLoss()
    
# #     model.to(device)
            
# #     valid_test_epoch_v2(name='Valid',model=model, dataset = dataset, device=device,data_loader=valid_loader,criterion=criterion, healthy = args.healthy)



# import pandas as pd

# df = pd.read_csv('/home/santosh.sanjeev/san_data_v2/metadata_v3/test_main_file_v2.csv', header=None)

# list_row = []
# for _, row in df.iterrows():
#     if row[8] == 1 and row[7] == 2:
#         list_row.append(row)

# rrow = []
# for row in list_row:
#     p = row[1].split('-')[-1]
#     rrow.append(p)

# print(len(rrow))
# print(len(set(rrow)))




# import os
# import pickle
# import pprint
# import random
# from glob import glob
# from os.path import exists, join
# from tqdm import tqdm
# import numpy as np
# import torch
# import sklearn.metrics
# from sklearn.metrics import roc_auc_score, accuracy_score
# import sklearn, sklearn.model_selection



# def train(model, dataset, train_loader, valid_loader, device, args):    

#     print(args.output_dir)

#     if not exists(args.output_dir):
#         os.makedirs(args.output_dir)

#     dataset_name = 'MIMIC-densenet'
#     # Optimizer
#     optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
#     print(optim)

#     criterion = torch.nn.BCEWithLogitsLoss()

#     # Checkpointing
#     start_epoch = 0
#     best_metric = 0.
#     weights_for_best_validauc = None
#     auc_test = None
#     metrics = []
#     weights_files = glob(join(args.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
#     if len(weights_files):
#         # Find most recent epoch
#         epochs = np.array(
#             [int(w[len(join(args.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
#         start_epoch = epochs.max()
#         weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
#         model.load_state_dict(torch.load(weights_file).state_dict())

#         with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
#             metrics = pickle.load(f)

#         best_metric = metrics[-1]['best_metric']
#         weights_for_best_validauc = model.state_dict()

#         print("Resuming training at epoch {0}.".format(start_epoch))
#         print("Weights loaded: {0}".format(weights_file))

#     model.to(device)
    
#     for epoch in range(start_epoch, args.n_epochs):

#         avg_loss = train_epoch(cfg=args,
#                                epoch=epoch,
#                                model=model,
#                                device=device,
#                                optimizer=optim,
#                                train_loader=train_loader,
#                                criterion=criterion)
        
#         auc_valid = valid_test_epoch(name='Valid',
#                                      epoch=epoch,
#                                      model=model,
#                                      device=device,
#                                      data_loader=valid_loader,
#                                      criterion=criterion)[0]

#         if np.mean(auc_valid) > best_metric:
#             best_metric = np.mean(auc_valid)
#             weights_for_best_validauc = model.state_dict()
#             torch.save(model, join(args.output_dir, f'{dataset_name}-best.pt'))
#             # only compute when we need to

#         stat = {
#             "epoch": epoch + 1,
#             "trainloss": avg_loss,
#             "validauc": auc_valid,
#             'best_metric': best_metric
#         }

#         metrics.append(stat)

#         with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
#             pickle.dump(metrics, f)

#         torch.save(model, join(args.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

#     return metrics, best_metric, weights_for_best_validauc





# def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, limit=None):
#     model.train()

#     if cfg.taskweights:
#         weights = np.nansum(train_loader.dataset.labels, axis=0)
#         weights = weights.max() - weights + weights.mean()
#         weights = weights/weights.max()
#         weights = torch.from_numpy(weights).to(device).float()
#         print("task weights", weights)
    
#     avg_loss = []
#     t = tqdm(train_loader)
#     for batch_idx, samples in enumerate(t):
        
#         if limit and (batch_idx > limit):
#             print("breaking out")
#             break
            
#         optimizer.zero_grad()
        
#         images = samples["img"].float().to(device)
#         targets = samples["lab"].to(device)
#         outputs = model(images)
        
#         loss = torch.zeros(1).to(device).float()
#         for task in range(targets.shape[1]):
#             task_output = outputs[:,task]
#             task_target = targets[:,task]
#             mask = ~torch.isnan(task_target)
#             task_output = task_output[mask]
#             task_target = task_target[mask]
#             if len(task_target) > 0:
#                 task_loss = criterion(task_output.float(), task_target.float())
#                 if cfg.taskweights:
#                     loss += weights[task]*task_loss
#                 else:
#                     loss += task_loss
        
#         # here regularize the weight matrix when label_concat is used
#         if cfg.label_concat_reg:
#             if not cfg.label_concat:
#                 raise Exception("cfg.label_concat must be true")
#             weight = model.classifier.weight
#             num_labels = 13
#             num_datasets = weight.shape[0]//num_labels
#             weight_stacked = weight.reshape(num_datasets,num_labels,-1)
#             label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
#             for task in range(num_labels):
#                 dists = torch.pdist(weight_stacked[:,task], p=2).mean()
#                 loss += label_concat_reg_lambda*dists
                
#         loss = loss.sum()
        
#         if cfg.featurereg:
#             feat = model.features(images)
#             loss += feat.abs().sum()
            
#         if cfg.weightreg:
#             loss += model.classifier.weight.abs().sum()
        
#         loss.backward()

#         avg_loss.append(loss.detach().cpu().numpy())
#         t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

#         optimizer.step()

#     return np.mean(avg_loss)

# def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
#     model.eval()

#     avg_loss = []
#     task_outputs={}
#     task_targets={}
#     for task in range(data_loader.dataset[0]["lab"].shape[0]):
#         task_outputs[task] = []
#         task_targets[task] = []
        
#     with torch.no_grad():
#         t = tqdm(data_loader)
#         for batch_idx, samples in enumerate(t):

#             if limit and (batch_idx > limit):
#                 print("breaking out")
#                 break
            
#             images = samples["img"].to(device)
#             targets = samples["lab"].to(device)

#             outputs = model(images)
            
#             loss = torch.zeros(1).to(device).double()
#             for task in range(targets.shape[1]):
#                 task_output = outputs[:,task]
#                 task_target = targets[:,task]
#                 mask = ~torch.isnan(task_target)
#                 task_output = task_output[mask]
#                 task_target = task_target[mask]
#                 if len(task_target) > 0:
#                     loss += criterion(task_output.double(), task_target.double())
                
#                 task_outputs[task].append(task_output.detach().cpu().numpy())
#                 task_targets[task].append(task_target.detach().cpu().numpy())

#             loss = loss.sum()
            
#             avg_loss.append(loss.detach().cpu().numpy())
#             t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
#         for task in range(len(task_targets)):
#             task_outputs[task] = np.concatenate(task_outputs[task])
#             task_targets[task] = np.concatenate(task_targets[task])
    
#         task_aucs = []
#         for task in range(len(task_targets)):
#             if len(np.unique(task_targets[task]))> 1:
#                 task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
#                 #print(task, task_auc)
#                 task_aucs.append(task_auc)
#             else:
#                 task_aucs.append(np.nan)

#     task_aucs = np.asarray(task_aucs)
#     auc = np.mean(task_aucs[~np.isnan(task_aucs)])
#     print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

#     return auc, task_aucs, task_outputs, task_targets


# def valid_test_epoch_v2(name, dataset, model, device, data_loader, criterion, healthy, limit=None):
#     model.eval()

#     avg_loss = []
#     task_outputs={}
#     task_targets={}
#     for task in range(13):
#         task_outputs[task] = []
#         task_targets[task] = []
        
#     with torch.no_grad():
#         t = tqdm(data_loader)
#         for batch_idx, samples in enumerate(t):

#             if limit and (batch_idx > limit):
#                 print("breaking out")
#                 break
#             # print('heloooooooooo')
#             images = samples["img"].to(device)
#             targets = samples["lab"].to(device)

#             outputs = model(images)
#             # outputs = samples['pred'].to(device)
#             # print(outputs)
#             loss = torch.zeros(1).to(device).double()
#             for task in range(targets.shape[1]):
#                 task_output = outputs[:,task]
#                 task_target = targets[:,task]
#                 mask = ~torch.isnan(task_target)
#                 task_output = task_output[mask]
#                 task_target = task_target[mask]
#                 if len(task_target) > 0:
#                     loss += criterion(task_output.double(), task_target.double())
                
#                 task_outputs[task].append(task_output.detach().cpu().numpy())
#                 task_targets[task].append(task_target.detach().cpu().numpy())

#             loss = loss.sum()
            
#             avg_loss.append(loss.detach().cpu().numpy())
#             t.set_description(f'Loss = {np.mean(avg_loss):4.4f}')
#             # print('heloooooooooo')

#         for task in range(len(task_targets)):
#             task_outputs[task] = np.concatenate(task_outputs[task])
#             task_targets[task] = np.concatenate(task_targets[task])
    
#         task_aucs = []
#         for task in range(len(task_targets)):
#             if len(np.unique(task_targets[task]))> 1:
#                 task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
#                 #print(task, task_auc)
#                 task_aucs.append(task_auc)
#             else:
#                 task_aucs.append(np.nan)
#         # print('heloooooooooo')

#     task_aucs = np.asarray(task_aucs)
#     auc = np.mean(task_aucs[~np.isnan(task_aucs)])
#     print(f'Avg AUC = {auc:4.4f}')
    
#     results = [auc, task_aucs, task_outputs, task_targets]
#     # all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]

#     # if healthy == True :
#     #     # print('hiiiiiiiii')
#     #     # [0.5938055, 0.5748906, 0.30114534, 0.18698719, 0.12614557, 0.04757202, 0.092572555, 0.5241535, 0.63551843, 0.028975938, 0.22440709, 0.09897159, 0.6400715]
#     #     all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]
#     #     all_min = []
#     #     all_max = []
#     #     all_ppv80 = []
#     #     all_accuracy = []
#     #     all_f1_score = []
#     #     all_precision = []
#     #     all_recall = []
#     #     all_auc = []
#     #     ls = []
#     #     # print(results)
#     #     for i, patho in enumerate(dataset.pathologies):
#     #         print(i, patho)
#     #         opt_thres = np.nan
#     #         opt_min = np.nan
#     #         opt_max = np.nan
#     #         ppv80_thres = np.nan
#     #         accuracy = np.nan
#     #         f1_score = np.nan
#     #         precision = np.nan
#     #         recall = np.nan
#     #         auc = np.nan
            
#     #         # if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
                
#     #         #sigmoid
#     #         all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
            
#     #         # fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
#     #         # pente = tpr - fpr
#     #         # opt_thres = thres_roc[np.argmax(pente)]
#     #         # opt_min = all_outputs.min()
#     #         # opt_max = all_outputs.max()
            
#     #         # ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
#     #         # ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
#     #         # ppv80_thres = thres_pr[ppv80_thres_idx-1]
            
#     #         # auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
            
#     #         # Calculate confusion matrix for accuracy, precision, recall, and F1 score
#     #         threshold = all_threshs[i]  
#     #         predicted_labels = (all_outputs >= threshold).astype(int)
#     #         true_labels = results[3][i]
#     #         print(patho)
#     #         print(predicted_labels)
#     #         print(true_labels)
#     #         ls.append(predicted_labels)
#     #     array = np.array(ls)
#     #     arr2 = array.T
#     #     print(arr2, arr2.shape)

#     #     rows_with_all_zeros = np.sum(np.all(arr2==0,axis=1))
#     #     print(rows_with_all_zeros)
            

#     perf_dict = {}
#     # all_threshs = [1.305559408137924e-06, 7.167565740928694e-08, 2.975418356143677e-09, 1.1836777957796585e-06, 2.679294297536217e-08, np.nan, 5.104131517441601e-08, 1.6621943643713166e-07, 2.5561273986340893e-08, 6.293478238550421e-11, 3.9989647149241137e-08, 5.29030330653768e-09, 6.500377480733732e-08]
#     # all_threshs = [0.2803523540496826, 0.9998441934585572, 0.9989217519760132, 0.0319266691803932, 0.999996781349182, np.nan, 1.0, 0.9993218183517456, 0.9999105930328368, 1.0, 9.052349196281284e-05, 0.0214367806911468, 0.9984620809555054]
#     # all_threshs = [0.7227323739299079, 0.7306809896548937, 0.7228490057843762, 0.5052116874830279, 0.7310508909067585, np.nan, 0.7310585786300049, 0.7308803893342449, 0.7310429686563822, 0.7310585551920353, 0.7123462598961197, 0.6589979007539639, 0.7298141312156243]#[]
#     # all_threshs = [0.5938055, 0.5748906, 0.30114534, 0.18698719, 0.12614557, 0.04757202, 0.092572555, 0.5241535, 0.63551843, 0.028975938, 0.22440709, 0.09897159, 0.6400715]
#     all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]
#     all_min = []
#     all_max = []
#     all_ppv80 = []
#     all_accuracy = []
#     all_f1_score = []
#     all_precision = []
#     all_recall = []
#     all_auc = []
#     # print(results)
#     for i, patho in enumerate(dataset.pathologies):
#         print(i, patho)
#         opt_thres = np.nan
#         opt_min = np.nan
#         opt_max = np.nan
#         ppv80_thres = np.nan
#         accuracy = np.nan
#         f1_score = np.nan
#         precision = np.nan
#         recall = np.nan
#         auc = np.nan
        
#         if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
            
#             #sigmoid
#             # print(results[2][i])
#             all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
#             # print(all_outputs)
            
#             fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
#             pente = tpr - fpr
#             opt_thres = thres_roc[np.argmax(pente)]
#             opt_min = all_outputs.min()
#             opt_max = all_outputs.max()
            
#             ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
#             ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
#             ppv80_thres = thres_pr[ppv80_thres_idx-1]
            
#             auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
            
#             # Calculate confusion matrix for accuracy, precision, recall, and F1 score
#             # threshold = opt_thres#all_threshs[i]
#             threshold = all_threshs[i]  
#             predicted_labels = (all_outputs >= threshold).astype(int)
#             true_labels = results[3][i]
#             confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
#             TP = confusion_matrix[1, 1]
#             TN = confusion_matrix[0, 0]
#             FP = confusion_matrix[0, 1]
#             FN = confusion_matrix[1, 0]

#             # Calculate metrics
#             accuracy = (TP + TN) / (TP + TN + FP + FN)
#             precision = TP / (TP + FP)
#             recall = TP / (TP + FN)
#             f1_score = 2 * (precision * recall) / (precision + recall)
            
#             # Add metrics to perf_dict
#             perf_dict[patho] = {
#                 'AUC': round(auc, 2),
#                 'Accuracy': round(accuracy, 2),
#                 'F1 Score': round(f1_score, 2),
#                 'Precision': round(precision, 2),
#                 'Recall': round(recall, 2)
#             }
            
#             all_auc.append(auc)  # Append AUC to the list
            
#         else:
#             perf_dict[patho] = "-"
    



#         # Append metrics to respective lists
#         all_threshs.append(opt_thres)
#         all_min.append(opt_min)
#         all_max.append(opt_max)
#         all_ppv80.append(ppv80_thres)
#         all_accuracy.append(accuracy)
#         all_f1_score.append(f1_score)
#         all_precision.append(precision)
#         all_recall.append(recall)


#     # for i in enumerate(len(task_targets)):
#     #     c=0
#     #     for  j,patho in enumerate(dataset.pathologies):
#     #         if task_targets[i][j] != 0:
#     #             c=1
#     #             break
#     #     if c==0:
            

#         # print(i, patho)
#         # opt_thres = np.nan
#         # opt_min = np.nan
#         # opt_max = np.nan
#         # ppv80_thres = np.nan
#         # accuracy = np.nan
#         # f1_score = np.nan
#         # precision = np.nan
#         # recall = np.nan
#         # auc = np.nan
        
#         # if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):





#     # Print the results
#     print("pathologies", dataset.pathologies)
#     print("------------------------------------------------------------------------------------------------")
#     print("op_threshs", str(all_threshs).replace("nan", "np.nan"))
#     print("min", str(all_min).replace("nan", "np.nan"))
#     print("max", str(all_max).replace("nan", "np.nan"))
#     print("ppv80", str(all_ppv80).replace("nan", "np.nan"))
#     print("accuracy", str(all_accuracy).replace("nan", "np.nan"))
#     print("f1_score", str(all_f1_score).replace("nan", "np.nan"))
#     print("precision", str(all_precision).replace("nan", "np.nan"))
#     print("recall", str(all_recall).replace("nan", "np.nan"))
#     print("all AUC values:", str(all_auc).replace("nan", "np.nan"))

#     # Calculate and print average metrics
#     avg_accuracy = np.nanmean(all_accuracy)
#     avg_f1_score = np.nanmean(all_f1_score)
#     avg_precision = np.nanmean(all_precision)
#     avg_recall = np.nanmean(all_recall)
#     avg_auc = np.nanmean(all_auc)

#     print(f'Average Accuracy: {round(avg_accuracy, 2)}')
#     print(f'Average F1 Score: {round(avg_f1_score, 2)}')
#     print(f'Average Precision: {round(avg_precision, 2)}')
#     print(f'Average Recall: {round(avg_recall, 2)}')
#     print(f'Average AUC: {round(avg_auc, 2)}')
        





# def validate(model, checkpoint_path, dataset, valid_loader, device, args):
#     print('inside validate',len(dataset), valid_loader)
#     model.load_state_dict(torch.load(checkpoint_path).state_dict())
#     criterion = torch.nn.BCEWithLogitsLoss()
    
#     model.to(device)
            
#     valid_test_epoch_v2(name='Valid',model=model, dataset = dataset, device=device,data_loader=valid_loader,criterion=criterion, healthy = args.healthy)




import pandas as pd

# df = pd.read_csv('/home/santosh.sanjeev/san_data_v2/metadata_v3/test_main_file_v2.csv', header=None)

# list_row = []
# for _, row in df.iterrows():
#     if row[8] == 1 and row[7] == 2:
#         list_row.append(row)

# rrow = []
# for row in list_row:
#     p = row[1].split('-')[-1]
#     rrow.append(p)

# print(len(rrow))
# print(len(set(rrow)))




import os
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join
from tqdm import tqdm
import numpy as np
import torch
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import wandb


def train(model, dataset, train_loader, valid_loader, device, args):    
    # init wandb
    wandb.init(project="vindr-classification", config=args)
    

    print(args.output_dir)

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset_name = 'MIMIC-densenet'
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    print(optim)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(args.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(args.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)
    
    for epoch in range(start_epoch, args.n_epochs):

        avg_loss = train_epoch(cfg=args,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion)
        
        auc_valid = valid_test_epoch(name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion)[0]

        if np.mean(auc_valid) > best_metric:
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(args.output_dir, f'{dataset_name}-best.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        torch.save(model, join(args.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

    return metrics, best_metric, weights_for_best_validauc





def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, limit=None):
    model.train()

    if cfg.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights/weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights", weights)
    
    avg_loss = []
    t = tqdm(train_loader)
    for batch_idx, samples in enumerate(t):
        
        if limit and (batch_idx > limit):
            print("breaking out")
            break
            
        optimizer.zero_grad()
        
        images = samples["img"].float().to(device)
        targets = samples["lab"].to(device)
        outputs = model(images)
        
        loss = torch.zeros(1).to(device).float()
        for task in range(targets.shape[1]):
            task_output = outputs[:,task]
            task_target = targets[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = criterion(task_output.float(), task_target.float())
                if cfg.taskweights:
                    loss += weights[task]*task_loss
                else:
                    loss += task_loss
        
        # here regularize the weight matrix when label_concat is used
        if cfg.label_concat_reg:
            if not cfg.label_concat:
                raise Exception("cfg.label_concat must be true")
            weight = model.classifier.weight
            num_labels = 13
            num_datasets = weight.shape[0]//num_labels
            weight_stacked = weight.reshape(num_datasets,num_labels,-1)
            label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
            for task in range(num_labels):
                dists = torch.pdist(weight_stacked[:,task], p=2).mean()
                loss += label_concat_reg_lambda*dists
                
        loss = loss.sum()
        
        if cfg.featurereg:
            feat = model.features(images)
            loss += feat.abs().sum()
            
        if cfg.weightreg:
            loss += model.classifier.weight.abs().sum()
        
        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

        wandb.log({
            "Train/loss": loss.detach().cpu().numpy(),
            "epoch": epoch
        }, step=batch_idx + len(train_loader) * epoch)

    return np.mean(avg_loss)

def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            outputs = model(images)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')

            wandb.log({
                "Val/loss": loss.detach().cpu().numpy(),
            }, step=batch_idx + len(data_loader) * epoch)

        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

        wandb.log({
            "Val/AUC": np.mean(task_aucs),
        }, step=batch_idx + len(data_loader) * epoch)
        
    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return auc, task_aucs, task_outputs, task_targets


def valid_test_epoch_v2(name, dataset, model, device, data_loader, criterion, healthy, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(13):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break
            print('heloooooooooo')
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            outputs = model(images)#samples['pred'].to(device)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Loss = {np.mean(avg_loss):4.4f}')
            print('heloooooooooo')

        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)
        print('heloooooooooo')

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Avg AUC = {auc:4.4f}')
    
    results = [auc, task_aucs, task_outputs, task_targets]
    # all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]

    if healthy == True :
        print('hiiiiiiiii')
        # [0.5938055, 0.5748906, 0.30114534, 0.18698719, 0.12614557, 0.04757202, 0.092572555, 0.5241535, 0.63551843, 0.028975938, 0.22440709, 0.09897159, 0.6400715]
        all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]
        all_min = []
        all_max = []
        all_ppv80 = []
        all_accuracy = []
        all_f1_score = []
        all_precision = []
        all_recall = []
        all_auc = []
        ls = []
        # print(results)
        for i, patho in enumerate(dataset.pathologies):
            print(i, patho)
            opt_thres = np.nan
            opt_min = np.nan
            opt_max = np.nan
            ppv80_thres = np.nan
            accuracy = np.nan
            f1_score = np.nan
            precision = np.nan
            recall = np.nan
            auc = np.nan
            
            # if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
                
            #sigmoid
            all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
            
            # fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
            # pente = tpr - fpr
            # opt_thres = thres_roc[np.argmax(pente)]
            # opt_min = all_outputs.min()
            # opt_max = all_outputs.max()
            
            # ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
            # ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
            # ppv80_thres = thres_pr[ppv80_thres_idx-1]
            
            # auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
            
            # Calculate confusion matrix for accuracy, precision, recall, and F1 score
            threshold = all_threshs[i]  
            predicted_labels = (all_outputs >= threshold).astype(int)
            true_labels = results[3][i]
            print(patho)
            print(predicted_labels)
            print(true_labels)
            ls.append(predicted_labels)
        array = np.array(ls)
        arr2 = array.T
        print(arr2, arr2.shape)

        rows_with_all_zeros = np.sum(np.all(arr2==0,axis=1))
        print(rows_with_all_zeros)
            

    else:
        perf_dict = {}
        # all_threshs = [0.7227323739299079, 0.7306809896548937, 0.7228490057843762, 0.5052116874830279, 0.7310508909067585, np.nan, 0.7310585786300049, 0.7308803893342449, 0.7310429686563822, 0.7310585551920353, 0.7123462598961197, 0.6589979007539639, 0.7298141312156243]#[]
        # [0.5938055, 0.5748906, 0.30114534, 0.18698719, 0.12614557, 0.04757202, 0.092572555, 0.5241535, 0.63551843, 0.028975938, 0.22440709, 0.09897159, 0.6400715]
        all_threshs = [0.65824, 0.495423, 0.4475553, 0.32523003, 0.22015645, 0.20265268, 0.25850585, 0.5791169, 0.5172777, 0.041397225, 0.3125376, 0.115106925, 0.5741666]#[0.35617206, 0.24429399, 0.054616444, 0.11331658, 0.08341246, 0.044436485, 0.13542667, 0.40765527, 0.5090657, 0.041397225, 0.1737865, 0.115106925, 0.3480259]
        all_min = []
        all_max = []
        all_ppv80 = []
        all_accuracy = []
        all_f1_score = []
        all_precision = []
        all_recall = []
        all_auc = []
        # print(results)
        for i, patho in enumerate(dataset.pathologies):
            print(i, patho)
            opt_thres = np.nan
            opt_min = np.nan
            opt_max = np.nan
            ppv80_thres = np.nan
            accuracy = np.nan
            f1_score = np.nan
            precision = np.nan
            recall = np.nan
            auc = np.nan
            
            if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
                
                #sigmoid
                all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
                
                fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
                pente = tpr - fpr
                opt_thres = thres_roc[np.argmax(pente)]
                opt_min = all_outputs.min()
                opt_max = all_outputs.max()
                
                ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
                ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
                ppv80_thres = thres_pr[ppv80_thres_idx-1]
                
                auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
                
                # Calculate confusion matrix for accuracy, precision, recall, and F1 score
                threshold = all_threshs[i]  
                predicted_labels = (all_outputs >= threshold).astype(int)
                true_labels = results[3][i]
                confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
                TP = confusion_matrix[1, 1]
                TN = confusion_matrix[0, 0]
                FP = confusion_matrix[0, 1]
                FN = confusion_matrix[1, 0]

                # Calculate metrics
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1_score = 2 * (precision * recall) / (precision + recall)
                
                # Add metrics to perf_dict
                perf_dict[patho] = {
                    'AUC': round(auc, 2),
                    'Accuracy': round(accuracy, 2),
                    'F1 Score': round(f1_score, 2),
                    'Precision': round(precision, 2),
                    'Recall': round(recall, 2)
                }
                
                all_auc.append(auc)  # Append AUC to the list
                
            else:
                perf_dict[patho] = "-"
        



            # Append metrics to respective lists
            all_threshs.append(opt_thres)
            all_min.append(opt_min)
            all_max.append(opt_max)
            all_ppv80.append(ppv80_thres)
            all_accuracy.append(accuracy)
            all_f1_score.append(f1_score)
            all_precision.append(precision)
            all_recall.append(recall)


        # for i in enumerate(len(task_targets)):
        #     c=0
        #     for  j,patho in enumerate(dataset.pathologies):
        #         if task_targets[i][j] != 0:
        #             c=1
        #             break
        #     if c==0:
                

            # print(i, patho)
            # opt_thres = np.nan
            # opt_min = np.nan
            # opt_max = np.nan
            # ppv80_thres = np.nan
            # accuracy = np.nan
            # f1_score = np.nan
            # precision = np.nan
            # recall = np.nan
            # auc = np.nan
            
            # if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):





        # Print the results
        print("pathologies", dataset.pathologies)
        print("------------------------------------------------------------------------------------------------")
        print("op_threshs", str(all_threshs).replace("nan", "np.nan"))
        print("min", str(all_min).replace("nan", "np.nan"))
        print("max", str(all_max).replace("nan", "np.nan"))
        print("ppv80", str(all_ppv80).replace("nan", "np.nan"))
        print("accuracy", str(all_accuracy).replace("nan", "np.nan"))
        print("f1_score", str(all_f1_score).replace("nan", "np.nan"))
        print("precision", str(all_precision).replace("nan", "np.nan"))
        print("recall", str(all_recall).replace("nan", "np.nan"))
        print("all AUC values:", str(all_auc).replace("nan", "np.nan"))

        # Calculate and print average metrics
        avg_accuracy = np.nanmean(all_accuracy)
        avg_f1_score = np.nanmean(all_f1_score)
        avg_precision = np.nanmean(all_precision)
        avg_recall = np.nanmean(all_recall)
        avg_auc = np.nanmean(all_auc)

        print(f'Average Accuracy: {round(avg_accuracy, 2)}')
        print(f'Average F1 Score: {round(avg_f1_score, 2)}')
        print(f'Average Precision: {round(avg_precision, 2)}')
        print(f'Average Recall: {round(avg_recall, 2)}')
        print(f'Average AUC: {round(avg_auc, 2)}')
            





def validate(model, checkpoint_path, dataset, valid_loader, device, args):
    print('inside validate',len(dataset), valid_loader)
    model.load_state_dict(torch.load(checkpoint_path).state_dict())
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.to(device)
            
    valid_test_epoch_v2(name='Valid',model=model, dataset = dataset, device=device,data_loader=valid_loader,criterion=criterion, healthy = args.healthy)



