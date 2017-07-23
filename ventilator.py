# coding=utf-8
"""
produce train data for query2vec model
Run like this:
python ventilator.py q2v_aksis_ventilator ./data/rawdata --worker-num 30 -ap *add.201704 -1 -ap *purchase.201704 -1 -ap *click.201704 0.8 -bs 128
"""

import os
import sys
from argparser.customArgType import FileType
from argparser.customArgAction import AppendTupleWithoutDefault
import argparse
import signal
from utils.log_util import Logger
import logbook
from config.config import project_dir, vocabulary_size


def parse_args():
    parser = argparse.ArgumentParser(description='Ventilator for query2vec')
    parser.add_argument('--log-file-name', default=os.path.join(project_dir, 'data/logs', 'aksis_data.log'),
                        type=FileType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('--metric-interval', default=6, type=int,
                        help='metric reporting frequency is set by seconds param')
    subparsers = parser.add_subparsers(help='Ventilator for query2vec train data')

    q2v_aksis_ventilator_parser = subparsers.add_parser("q2v_aksis_ventilator")
    q2v_aksis_ventilator_parser.set_defaults(action='q2v_aksis_ventilator')

    q2v_aksis_ventilator_parser.add_argument('--vocabulary-data-dir',
                                              default=os.path.join(project_dir, 'data', 'vocabulary'),
                                              type=str,
                                              help='path for vocabulary and word counter ')
    q2v_aksis_ventilator_parser.add_argument('--words-list-file',
                                              default=os.path.join(project_dir, 'data', 'vocabulary', 'words_list'),
                                              type=str,
                                              help='words list to build vocabulary')
    q2v_aksis_ventilator_parser.add_argument('-ap', '--action-patterns', nargs=2, action=AppendTupleWithoutDefault,
                                              default=[('*add', -1), ('*search', 0.5), ('*click', 0.4),
                                                       ('*purchase', -1)])
    q2v_aksis_ventilator_parser.add_argument('--ip-addr', type=str, help='ip address')
    q2v_aksis_ventilator_parser.add_argument('--port', type=str, help='zmq port')
    q2v_aksis_ventilator_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                                              help='batch size for each databatch')
    q2v_aksis_ventilator_parser.add_argument('--top-words', default=vocabulary_size, type=int,
                                              help='the max sample num for training')
    q2v_aksis_ventilator_parser.add_argument('--worker-num', default=1, type=int,
                                              help='number of parser worker')
    q2v_aksis_ventilator_parser.add_argument('--rawdata-frontend-port', default=5555, type=int,
                                              help='train rawdata frontend port')
    q2v_aksis_ventilator_parser.add_argument('--rawdata-backend-port', default=5556, type=int,
                                             help='train rawdata backend port')
    q2v_aksis_ventilator_parser.add_argument('--collector-frontend-port', default=5557, type=int,
                                             help='collector frontend port')
    q2v_aksis_ventilator_parser.add_argument('--collector-backend-port', default=5558, type=int,
                                             help='collector frontend port')
    return parser.parse_args()


def signal_handler(signal, frame):
    logbook.warn('Stop!!!')
    sys.exit(0)


def setup_logger():
    log = Logger()
    log.set_stream_handler()
    log.set_time_rotating_file_handler(args.log_file_name)


if __name__ == "__main__":
    args = parse_args()
    setup_logger()
    signal.signal(signal.SIGINT, signal_handler)
    if args.action == 'q2v_aksis_ventilator':
        from data_io.distribute_stream.aksis_data_pipeline import AksisDataPipeline

        p = AksisDataPipeline(args.vocabulary_data_dir, args.top_words, args.action_patterns,
                              args.batch_size, rawdata_frontend_port=args.rawdata_frontend_port,
                              rawdata_backend_port=args.rawdata_backend_port,
                              collector_fronted_port=args.collector_frontend_port,
                              collector_backend_port=args.collector_backend_port,
                              worker_num=args.worker_num, words_list_file=args.words_list_file)
        p.start_all()
