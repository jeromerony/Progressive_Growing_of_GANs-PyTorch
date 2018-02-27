# -*- coding: utf-8 -*-


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100,
                     fill='=', empty=' ', tip='>', begin='[', end=']', done="[DONE]", clear=True):
    """
    Print iterations progress.
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required : current iteration                          [int]
        total       - Required : total iterations                           [int]
        prefix      - Optional : prefix string                              [str]
        suffix      - Optional : suffix string                              [str]
        decimals    - Optional : positive number of decimals in percent     [int]
        length      - Optional : character length of bar                    [int]
        fill        - Optional : bar fill character                         [str] (ex: 'â– ', 'â–ˆ', '#', '=')
        empty       - Optional : not filled bar character                   [str] (ex: '-', ' ', 'â€¢')
        tip         - Optional : character at the end of the fill bar       [str] (ex: '>', '')
        begin       - Optional : starting bar character                     [str] (ex: '|', 'â–•', '[')
        end         - Optional : ending bar character                       [str] (ex: '|', 'â–', ']')
        done        - Optional : display message when 100% is reached       [str] (ex: "[DONE]")
        clear       - Optional : display completion message or leave as is  [str]
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength
    if iteration != total:
        bar = bar + tip
    bar = bar + empty * (length - filledLength - len(tip))
    display = '\r{prefix}{begin}{bar}{end} {percent}%{suffix}' \
        .format(prefix=prefix, begin=begin, bar=bar, end=end, percent=percent, suffix=suffix)
    print(display, end=''),  # comma after print() required for python 2
    if iteration == total:  # print with newline on complete
        if clear:  # display given complete message with spaces to 'erase' previous progress bar
            finish = '\r{prefix}{done}'.format(prefix=prefix, done=done)
            if hasattr(str, 'decode'):  # handle python 2 non-unicode strings for proper length measure
                finish = finish.decode('utf-8')
                display = display.decode('utf-8')
            clear = ' ' * max(len(display) - len(finish), 0)
            print(finish + clear)
        else:
            print('')
