#!/usr/bin/env node
// Same example for .ts (TypeScript)

let globalRunCount: number = 0;
let globalList: string[] = [];

function foo(): void {
    const arg1: string = "a";
    bar(globalList);
}

function bar(myList: string[]): void {
    globalRunCount = globalRunCount + 1;
    if (globalRunCount < 2) {
        bar(myList);
    }
    baz();
    console.log("hello");
    baz();
}

function baz(): void {
    console.log("world %s", globalList);
}

// main program
globalRunCount = 0;
globalList = ["a", "b"];
foo();
baz();
