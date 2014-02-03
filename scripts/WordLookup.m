function index = WordLookup(InputString)
global wordMap
if wordMap.isKey(InputString)
    index = wordMap(InputString);
else
    % Change Notice 11/7/11 by Brody: this was orignially index = 1, but
    % was changed to map unknowns to UNK instead of PADDING
    index = wordMap('UNK');
end
