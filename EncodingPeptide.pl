#! /usr/bin/perl


my $workdir="./";  #set up the work path where data files (blosum62.csv) are saved 

#all 20 types of amino acids
@aminolist = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y");

#Get Hashtable containing blosum62 matrix scores
%ht_blosum62 = GetBlosum62();
#print("matrix score\n");
# print "$_ $ht_blosum62{$_}\n" for (keys %ht_blosum62);

#the input epitope sequence that need to be encoded
#$line="SKERAKWLE" ;
$line="SK";

#use sliding window (2 amico acid) and calculate similarity scores
for(my $index=0; $index<length($line)-1; $index++){
    my $two=substr($line, $index, 2); #get two amino acid
	print("AA: $two\n");
    #calculate similarity score with all 20x20 combinations of two amino acids
    for(my $i=0; $i<=$#aminolist; $i++) {
        for(my $j=0; $j<=$#aminolist; $j++) {
			print("outer: $aminolist[$i] inner: $aminolist[$j]\n");
            my $mykey = GetKeyInHtBlosum62( substr($two,0,1), $aminolist[$i]);
            my $score1 = $ht_blosum62{$mykey};
			print("key1: $mykey, score1: $score1\n");
            $mykey = GetKeyInHtBlosum62( substr($two,1,1), $aminolist[$j]);
            my $score2 = $ht_blosum62{$mykey};
			print("key1: $mykey, score1: $score2\n");
            my $total_score=$score1+$score2; # this the score for one combination
            print ("$aminolist[$i]$aminolist[$j]\t$two\t$total_score,\n ");
        }
     }
}
print ("\n");
#use all total of 3200 total scores as a feature vector
  

#Get Hashtable contains blosum62 matrix scores
sub GetBlosum62 {
	open(IN_BLOSUM62, $workdir."blosum62.csv") or die ("Can't open < input.txt: $!");
	
	#The first line is seemed as the Head of blosum62 matrix
	$isHead = 1;

	while(<IN_BLOSUM62>) {
		chomp;
		$line1 = $_;
		$line = trim($line1);

		if($isHead) {
			@head = split(/,/, $line);
			
			$isHead = 0;
			next;
		}

		@collection = split(/,/, $line);

		for($i=0; $i<=$#head; $i++) {
			$str_key = GetKeyInHtBlosum62($collection[0], $head[$i]);
			if(!$temphash{$str_key}) {
				$temphash{$str_key} = $collection[$i + 1];
			}
			print("str_key: $str_key, score: $temphash{$str_key}\n")
		}
	}

	close(IN_BLOSUM62);
	return %temphash;

}


# Create key for hashtable blosum62
# key is composed of two amino acid characters, always in alphabetical order
sub GetKeyInHtBlosum62 {
	my($residue1, $residue2) = @_;

	if($residue1 gt $residue2) {
		return $residue2 . $residue1;
	}

	return $residue1 . $residue2;
}


sub trim($) {
	my $string = shift;
	$string =~ s/^\s+//;
	$string =~ s/\s+$//;
	return $string;
}


