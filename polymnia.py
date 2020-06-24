import optparse

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--length", action="store", help="text length", type='int')
    parser.add_option("--order", action="store", help="markov order", type="int")
    parser.add_option("--text", action="store", help="text body")

    (params, _) = parser.parse_args(sys.argv)

    generateWords(params.order, params.length, params.text)

