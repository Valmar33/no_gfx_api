
package main

import "core:fmt"
import vmem "core:mem/virtual"
import "core:mem"
import "core:strings"

codegen :: proc(ast: Ast)
{
    write_preamble()

    arena_backing: vmem.Arena
    ok_a := vmem.arena_init_growing(&arena_backing)
    assert(ok_a == nil)
    codegen_arena := vmem.arena_allocator(&arena_backing)
    defer free_all(codegen_arena)

    context.allocator = codegen_arena

    for declaration in ast.scope.decls
    {
        switch decl in declaration.derived_decl
        {
            case ^Ast_Struct_Decl:
            {
                writefln("struct %v", decl.name)
                writeln("{")
                if writer_scope()
                {
                    for field in decl.fields
                    {
                        writefln("%v %v;", type_to_glsl(field.type), field.name)
                    }
                }
                writeln("};")
                writeln("")
            }
            case ^Ast_Proc_Decl:
            {
                write_begin("")
                writef("%v %v(", type_to_glsl(decl.return_type), decl.name)
                for arg, i in decl.args
                {
                    writef("%v %v", type_to_glsl(arg.type), arg.name)
                    if i < len(decl.args) - 1 {
                        write(", ")
                    }
                }
                writeln(")")
                writeln("{")
                if writer_scope()
                {
                    for statement in decl.statements
                    {
                        codegen_statement(statement)
                    }
                }
                writeln("}")
                writeln("")
            }
            case:
            {
                fmt.println("Error!")
            }
        }
    }
}

codegen_statement :: proc(statement: ^Ast_Statement)
{
    write_begin("")

    switch stmt in statement.derived_statement
    {
        case ^Ast_Stmt_Expr:
        {
            codegen_expr(stmt.expr)
        }
        case ^Ast_Assign:
        {
            codegen_expr(stmt.lhs)
            write(" = ")
            codegen_expr(stmt.rhs)
        }
        case ^Ast_Var_Decl:
        {
            writef("%v %v", type_to_glsl(stmt.type), stmt.name)
        }
        case ^Ast_Return:
        {
            write("return ")
            codegen_expr(stmt.expr)
        }
    }

    write(";\n")
}

codegen_expr :: proc(expression: ^Ast_Expr)
{
    switch expr in expression.derived_expr
    {
        case ^Ast_Binary_Expr:
        {
            codegen_expr(expr.lhs)
            write(expr.token.text)
            codegen_expr(expr.rhs)
        }
        case ^Ast_Ident_Expr:
        {
            write(expr.token.text)
        }
        case ^Ast_Lit_Expr:
        {
            write(expr.token.text)
        }
        case ^Ast_Member_Access:
        {
            codegen_expr(expr.target)
            writef(".%v", expr.member_name)
        }
        case ^Ast_Array_Access:
        {
            codegen_expr(expr.target)
            write("[")
            codegen_expr(expr.idx_expr)
            write("]")
        }
        case ^Ast_Call:
        {
            codegen_expr(expr.target)
            write("(")
            for arg, i in expr.args
            {
                codegen_expr(arg)
                if i < len(expr.args) - 1 {
                    write(", ")
                }
            }
            write(")")
        }
    }
}

type_to_glsl :: proc(type: ^Ast_Type) -> string
{
    to_concat: []string
    if type.is_ptr || type.is_slice {
        to_concat = { "_res_ptr_", type.name }
    } else {
        to_concat = { type.name }
    }
    concatenated := strings.concatenate(to_concat)
    return concatenated
}

Writer :: struct
{
    indentation: u32,
}

@(private="file")
writer: Writer

@(deferred_in = writer_scope_end)
writer_scope :: proc() -> bool
{
    writer_scope_begin()
    return true
}

writer_scope_begin :: proc()
{
    writer.indentation += 1
}

writer_scope_end :: proc()
{
    writer.indentation -= 1
}

write_preamble :: proc()
{
    writeln("#version 460")
    writeln("#extension GL_EXT_buffer_reference : require")
    writeln("#extension GL_EXT_buffer_reference2 : require")
    writeln("")
}

writefln :: proc(fmt_str: string, args: ..any)
{
    write_indentation()
    fmt.printfln(fmt_str, ..args)
}

writef :: proc(fmt_str: string, args: ..any)
{
    fmt.printf(fmt_str, ..args)
}

writeln :: proc(strings: ..any)
{
    write_indentation()
    fmt.println(..strings)
}

write_begin :: proc(strings: ..any)
{
    write_indentation()
    fmt.print(..strings)
}

write :: proc(strings: ..any)
{
    fmt.print(..strings)
}

write_indentation :: proc()
{
    for i in 0..<4*writer.indentation {
        fmt.print(" ")
    }
}
