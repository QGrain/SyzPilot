import argparse, os, json
import matplotlib.pyplot as plt


def check_dir(d):
    if os.path.isdir(d):
        return True
    else:
        try:
            os.makedirs(d)
            return True
        except:
            return False


def sec2time(s):
    '''
    input: s, int: YYY
    return: t, str: XdXhXmXs
    '''
    days, seconds = divmod(s, 24 * 3600)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    time_str = ''
    if days > 0:
        time_str += f'{days}d'
    if hours > 0:
        time_str += f'{hours}h'
    if minutes > 0:
        time_str += f'{minutes}m'
    if seconds > 0 or not time_str:
        time_str += f'{seconds}s'

    return time_str

def time2sec(t):
    '''
    input: t, str: XdXhXmXs
    return: s, int: YYY
    '''
    if t.isdigit():
        return int(t)

    days, hours, minutes, seconds = 0, 0, 0, 0
    if 'd' in t:
        days = int(t.split('d')[0])
        t = t.split('d')[1]
    if 'h' in t:
        hours = int(t.split('h')[0])
        t = t.split('h')[1]
    if 'm' in t:
        minutes = int(t.split('m')[0])
        t = t.split('m')[1]
    if 's' in t:
        seconds = int(t.split('s')[0])

    total_seconds = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
    return total_seconds

def read_log(log_path):
    log_list = []
    with open(log_path, 'r', encoding='utf-8') as f:
        log_chunk = ''
        for line in f:
            log_chunk += line.strip()
            if log_chunk.endswith('}'):
                try:
                    log_dict = json.loads(log_chunk)
                    log_list.append(log_dict)
                    log_chunk = ''
                except json.JSONDecodeError as e:
                    print('[x] Errot parsing log_chunk to JSON: %s'%log_chunk)
                    log_chunk = ''
    return log_list

def search_log2(log_list, t):
    seconds = time2sec(t)
    print('[DEBUG] seconds = %d'%seconds)
    print('[DEBUG] log[0][time] = %d, log[-1][time] = %d, len(log) = %d'%(log_list[0]['uptime'], log_list[-1]['uptime'], len(log_list)))
    d = (log_list[-1]['uptime']-log_list[0]['uptime']) / len(log_list)
    n = int(seconds / d)
    l, r = max(0, n-5), min(len(log_list)-1, n+5)
    idx, min_delta_time = 0, 600
    for i in range(l, r):
        log_chunk = log_list[i]
        print('[DEBUG] log_list[%d][\'uptime\'] = %s, type = %s'%(i, log_chunk['uptime'], type(log_chunk['uptime'])))
        delta_time = abs(int(log_chunk['uptime'])-seconds)
        if delta_time < min_delta_time:
            idx = i
            min_delta_time = delta_time
    return log_list[idx]

def search_log(log_list, t):
    seconds = time2sec(t)
    d = (log_list[-1]['uptime']-log_list[0]['uptime']) / len(log_list)
    n = int(seconds / d)
    n = max(0, n)
    n = min(len(log_list)-1, n)
    i = n
    if log_list[i]['uptime'] < seconds:
        while i < len(log_list)-1:
            i += 1
            if abs(log_list[i]['uptime']-seconds) > abs(log_list[i-1]['uptime']-seconds):
                return i, log_list[i-1]
    else:
        while i > 0:
            i -= 1
            if abs(log_list[i]['uptime']-seconds) > abs(log_list[i+1]['uptime']-seconds):
                return i, log_list[i+1]
    return i, log_list[i]

def print_log(log_chunk, key=None):
    uptime_sec = log_chunk['uptime']
    uptime_time = sec2time(uptime_sec)
    print('[+] print log_chunk')
    print('    uptime: %d, %s'%(uptime_sec, uptime_time))
    if key != None and key != []:
        for k in key:
            print('    %s: %s'%(k, log_chunk[k]))
    else:
        # by default
        default = ['coverage', 'corpus', 'crashes', 'crash types', 'exec total']
        for k in default:
            print('    %s: %s'%(k, log_chunk[k]))

def stat_log(log_list):
    num_chunk = len(log_list)
    print('[-] This bench file contains %d log chunk'%num_chunk)
    print_log(log_list[-1])

def plot(title, d, key, legend, time, out_dir):
    plt.figure(figsize=(16, 10))
    plt.title(title, fontsize=28)
    plt.xlabel('time', fontsize=24)
    if key == 'BBhitCnt':
        plt.ylabel('FuncHitCnt', fontsize=24)
    else:
        plt.ylabel(key, fontsize=24)

    # t_max = 0
    for i in legend:
        # x = [0+60*i for i in range(l)]
        plt.plot(d[i]['uptime'], d[i][key], label=i, linewidth=2.0)
        # plt.plot(x, d[i][key], label=i)
        # t_max = max(t_max, d[i]['uptime'][-1])
    plt.legend(fontsize=28)
    plt.grid()

    seconds = time2sec(time)
    n_ticks = 12
    x_ticks = [i*seconds//n_ticks for i in range(n_ticks+1)]
    x_label = [sec2time(i*seconds//n_ticks) for i in range(n_ticks+1)]
    plt.xticks(x_ticks, x_label, fontsize=24)
    plt.yticks(fontsize=24)

    # plt.show()
    plt.savefig(os.path.join(out_dir, '%s-%s-interval-60s.png'%(key, time)))
    plt.savefig(os.path.join(out_dir, '%s-%s-interval-60s.eps'%(key, time)), dpi=600, format='eps')
    # plt.clf()
    print('[+] plot() done for %s-%s'%(key, time))

def calc_avg(d, k):
    # calc average of uptime
    dt = 0
    num = 100000000
    for i in d:
        num = min(num, len(d[i]['uptime']))
        dt += ((d[i]['uptime'][-1]-d[i]['uptime'][0])/num)/len(d)
        # print('%s: dt = %.2f'%(i, (d[i]['uptime'][-1]-d[i]['uptime'][0])/num))
    # print('avg dt = %.2f'%dt)
    x = [int(0+dt*i) for i in range(num)]
    avg_d_k = []
    for j in range(num):
        avg_v = 0
        for i in d:
            avg_v += (d[i][k][j])/len(d)
        avg_d_k.append(avg_v)
    return x, avg_d_k


def plot_avg(title, dcmp, dlab, key, time, out_dir):
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel(key)

    x_cmp, avg_cmp_k = calc_avg(dcmp, key)
    x_lab, avg_lab_k = calc_avg(dlab, key)
    plt.plot(x_cmp, avg_cmp_k, label='cmp-avg')
    plt.plot(x_lab, avg_lab_k, label='lab-avg')

    plt.legend()

    seconds = time2sec(time)
    n_ticks = 12
    x_ticks = [i*seconds//n_ticks for i in range(n_ticks+1)]
    x_label = [sec2time(i*seconds//n_ticks) for i in range(n_ticks+1)]
    plt.xticks(x_ticks, x_label)

    # plt.show()
    plt.savefig(os.path.join(out_dir, '%s-%s-average.png'%(key, time)))
    # plt.clf()
    print('[+] plot() done for %s-%s-average'%(key, time))

def get_args():
    parser = argparse.ArgumentParser(description='Parse fuzzing results from bench files')
    parser.add_argument('-b', '--bench_file', type=str, nargs='*', help='bench file(s) to parse')
    parser.add_argument('-t', '--time', type=str, help='print the results around specified time')
    parser.add_argument('-k', '--keys', type=str, nargs='*', help='keys to parse')
    parser.add_argument('-l', '--legend', type=str, nargs='*', help='legends for multi bench files, work with -p')
    parser.add_argument('-a', '--average', type=int, help='average number, work with -p')
    parser.add_argument('-p', '--plot', action='store_true', help='plot, work with -b, -t, -k, -l')
    parser.add_argument('-s', '--stat', action='store_true', help='stat the bench log, this option is prior to the others')
    parser.add_argument('-o', '--out_dir', type=str, help='out dir to save the plot fig')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    keys = args.keys or []
    out_dir = args.out_dir or '.'
    check_dir(out_dir)

    if args.bench_file:
        bench_list = args.bench_file
    else:
        print('[x] You have to specify at least one bench file through -b/--bench_file')
        exit(1)

    if args.time:
        t = args.time
    else:
        print('[x] You\'d better specify a time through -t, use default: 12h')
        t = '12h'

    legend = args.legend or [os.path.basename(_) for _ in bench_list]
    data = {}

    for i, log_path in enumerate(bench_list):
        print('[+] Proccessing %s'%log_path)
        log_list = read_log(log_path)

        if args.stat == True:
            stat_log(log_list)
            continue

        idx, log_chunk = search_log(log_list, t)

        if args.plot == False:
            print_log(log_chunk, keys)
        else:
            data[legend[i]] = {'uptime':[]}
            for k in keys:
                data[legend[i]][k] = []
            for j in range(idx+1):
                data[legend[i]]['uptime'].append(log_list[j]['uptime'])
                for k in keys:
                    try:
                        data[legend[i]][k].append(log_list[j][k])
                    except:
                        data[legend[i]][k].append(0)

    if args.plot == True:
        for k in keys:
            if k == 'BBhitCnt':
                plot('Results of FuncHitCnt', data, k, legend, t, out_dir)
            else:
                plot('Results of %s'%k, data, k, legend, t, out_dir)
        if args.average:
            pass
    print('[+] bench_parser done!')


if __name__ == '__main__':
    main()
    # python3 bench_parser.py -b ../test/cmp1.log ../test/cmp2.log ../test/lab1.log ../test/lab2.log  -t 1d12h -k coverage corpus -l cmp1 cmp2 lab1 lab2 -p -o /path/to/save/