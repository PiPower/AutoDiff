class Expression
{
public:
    virtual void compile() = 0;
    virtual ~Expression();
protected:
    Expression() = default;
};
