import colorama

'''Used only as a visual indication for how long the algorithms need to run for'''

def progress_bar(progress,total,color=colorama.Fore.RED):

    percent = 100.0*progress/total

    bar = '█' *int(percent) + '-'*(100-int(percent))
    print(color + f'\r|{bar}| {percent:.2f}%',end = '\r')
    
    if progress == total:
        print(colorama.Fore.GREEN + f'\r|{bar}| {percent:.2f}%',end = '\r')
        print('')
        print(colorama.Fore.RESET)




if __name__ == '__main__':

    print(colorama.Fore.RESET)

    for i in range(1,1_000_01):
        progress_bar(i,1_000_00)




    print('end')
