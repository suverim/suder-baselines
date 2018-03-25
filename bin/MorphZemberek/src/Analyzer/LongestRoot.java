package Analyzer;

import zemberek.morphology.analysis.WordAnalysis;
import zemberek.morphology.analysis.tr.TurkishMorphology;

import java.io.IOException;
import java.util.List;

import java.io.File;
import java.util.Scanner;

import java.io.FileOutputStream;

import java.io.BufferedWriter;
import java.io.OutputStreamWriter;

public class LongestRoot {

    TurkishMorphology morphology;

    public LongestRoot(TurkishMorphology morphology) {
        this.morphology = morphology;
    }

    public String get_longest_root(String word) {

        List<WordAnalysis> results = morphology.analyze(word);

        int lennow = -1;
        String outnow = "";

        for (WordAnalysis result : results) {
            String rootnow = result.dictionaryItem.root;
            if (! rootnow.equals("UNK")) {
                if (rootnow.length() > lennow) {
                    lennow = rootnow.length();
                    outnow = rootnow;
                }
            }
        }

        if (lennow == -1){
            outnow = word;
        }

        return outnow;
    }

    public static void main(String[] args) throws IOException {

        String inputfile = args[0];
        String outfile = args[1];

        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();

        LongestRoot longestRoot = new LongestRoot(morphology);

        FileOutputStream fout = new FileOutputStream(new File(outfile));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fout));

        Scanner scan = new Scanner(new File(inputfile));

        while(scan.hasNextLine()){

            String line = scan.nextLine();
            line = line.replace("\n", "").replace("\r", "");

            String rootnow = longestRoot.get_longest_root(line);
            String outnow = line+" "+rootnow+"\n";

            bw.write(line+" "+rootnow);
            bw.newLine();

        }

        bw.close();
        fout.close();

    }

}
