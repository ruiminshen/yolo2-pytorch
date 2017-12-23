import os

import numpy
import humanize
import pybenchmark

lambda env, cls_ap: ('timestamp', float(env.now.timestamp()))
lambda env, cls_ap: ('time', env.now.strftime('%Y-%m-%d %H:%M:%S'))
lambda env, cls_ap: ('step', env.step)
lambda env, cls_ap: ('epoch', env.epoch)
lambda env, cls_ap: ('model', env.config.get('model', 'dnn'))
lambda env, cls_ap: ('size_dnn', humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in env.inference.state_dict().values())))
lambda env, cls_ap: ('time_inference', pybenchmark.stats['inference']['time'])
lambda env, cls_ap: ('root', os.path.basename(env.config.get('config', 'root')))
lambda env, cls_ap: ('cache_name', env.config.get('cache', 'name'))
lambda env, cls_ap: ('model_name', env.config.get('model', 'name'))
lambda env, cls_ap: ('category', env.config.get('cache', 'category'))
lambda env, cls_ap: ('dataset_size', len(env.loader.dataset))
lambda env, cls_ap: ('detect_threshold', env.config.getfloat('detect', 'threshold'))
lambda env, cls_ap: ('detect_overlap', env.config.getfloat('detect', 'overlap'))
lambda env, cls_ap: ('eval_iou', env.config.getfloat('eval', 'iou'))
lambda env, cls_ap: ('eval_mean_ap', numpy.mean(list(cls_ap.values())))
lambda env, cls_ap: ('eval_ap', ', '.join(['%s=%f' % (env.category[c], cls_ap[c]) for c in sorted(cls_ap.keys())]))
lambda env, cls_ap: ('hparam', ', '.join([option + '=' + value for option, value in env._config.items('hparam')]) if hasattr(env, '_config') else None)
